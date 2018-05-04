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

//#define OSL_DEV
//#define __OSL_TRACE_MASKS

#include <cmath>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"
#include "OSL/genclosure.h"
#include "backendllvm_wide.h"
// TODO:  remove if possible, having the here breaks original encapsulation
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace pvt {

static ustring op_and("and");
static ustring op_bitand("bitand");
static ustring op_bitor("bitor");
static ustring op_break("break");
static ustring op_ceil("ceil");
static ustring op_cellnoise("cellnoise");
static ustring op_color("color");
static ustring op_compl("compl");
static ustring op_continue("continue");
static ustring op_dowhile("dowhile");
static ustring op_eq("eq");
static ustring op_error("error");
static ustring op_fabs("fabs");
static ustring op_floor("floor");
static ustring op_for("for");
static ustring op_format("format");
static ustring op_ge("ge");
static ustring op_gt("gt");
static ustring op_if("if");
static ustring op_le("le");
static ustring op_logb("logb");
static ustring op_lt("lt");
static ustring op_min("min");
static ustring op_neq("neq");
static ustring op_normal("normal");
static ustring op_or("or");
static ustring op_point("point");
static ustring op_printf("printf");
static ustring op_round("round");
static ustring op_shl("shl");
static ustring op_shr("shr");
static ustring op_sign("sign");
static ustring op_step("step");
static ustring op_trunc("trunc");
static ustring op_vector("vector");
static ustring op_warning("warning");
static ustring op_xor("xor");

static ustring u_distance ("distance");
static ustring u_index ("index");
static ustring u__empty;  // empty/default ustring



/// Macro that defines the arguments to LLVM IR generating routines
///
#define LLVMGEN_ARGS     BackendLLVMWide &rop, int opnum

/// Macro that defines the full declaration of an LLVM generator.
///
#define LLVMGEN(name)  bool name (LLVMGEN_ARGS)

// Forward decl
LLVMGEN (llvm_gen_generic);


static const char * warg_lane_count(void)
{
    switch(SimdLaneCount)
    {
    case 4:
        return "w4";
        break;
    case 8:
        return "w8";
        break;
    case 16:
        return "w16";
        break;
    default:
        ASSERT(0);
    };
    return nullptr;
}

static std::string
warg_typecode (Symbol *sym, bool derivs)
{
    std::string name(warg_lane_count());

    const TypeSpec &t (sym->typespec());
    if (t.is_int())
        name += "i";
    else if (t.is_matrix())
        name += "m";
    else if (t.is_string())
        name += "s";
    else {
		if (derivs)
			name += "d";

		if (t.is_float())
			name += "f";
		else if (t.is_triple())
			name += "v";
		else ASSERT (0);
    }
    return name;
}

static std::string
arg_typecode (Symbol &sym, bool derivs, bool is_uniform)
{
    std::string name;
    if(!is_uniform) {
    	name = warg_lane_count();
    }

    const TypeSpec &t (sym.typespec());
    if (t.is_int())
        name += "i";
    else if (t.is_matrix())
        name += "m";
    else if (t.is_string())
        name += "s";
    else {

		if (derivs)
			name += "d";

		if (t.is_float())
			name += "f";
		else if (t.is_triple())
			name += "v";
		else ASSERT (0);
    }
    return name;
}


void
BackendLLVMWide::llvm_gen_debug_printf (string_view message)
{
	ASSERT(0 && "incomplete, unsure if callable");
    ustring s = ustring::format ("(%s %s) %s", inst()->shadername(),
                                 inst()->layername(), message);
    ll.call_function ("osl_printf", sg_void_ptr(), ll.constant("%s\n"),
                      ll.constant(s));
}



void
BackendLLVMWide::llvm_gen_warning (string_view message)
{
	ASSERT(0 && "incomplete, unsure if callable");
    ll.call_function ("osl_warning", sg_void_ptr(), ll.constant("%s\n"),
                      ll.constant(message));
}



void
BackendLLVMWide::llvm_gen_error (string_view message)
{
	ASSERT(0 && "incomplete, unsure if callable");
    ll.call_function ("osl_error", sg_void_ptr(), ll.constant("%s\n"),
                      ll.constant(message));
}



void
BackendLLVMWide::llvm_call_layer (int layer, bool unconditional)
{
    OSL_DEV_ONLY(std::cout << "llvm_call_layer layer=" <<layer<< " unconditional=" << unconditional << std::endl);
    // Make code that looks like:
    //     if (! groupdata->run[parentlayer])
    //         parent_layer (sg, groupdata);
    // if it's a conditional call, or
    //     parent_layer (sg, groupdata);
    // if it's run unconditionally.
    // The code in the parent layer itself will set its 'executed' flag.

    llvm::Value *args[3];
    args[0] = sg_ptr ();
    args[1] = groupdata_ptr ();

    ShaderInstance *parent = group()[layer];
    llvm::Value *layerfield = layer_run_ref(layer_remap(layer));
    llvm::BasicBlock *then_block = NULL, *after_block = NULL;
    llvm::Value *lanes_requiring_execution_value = nullptr;
    if (! unconditional) {
        llvm::Value *previously_executed = ll.int_as_mask(ll.op_load (layerfield));
        llvm::Value *lanes_requiring_execution = ll.op_select(previously_executed, ll.wide_constant_bool(false), ll.current_mask());
        lanes_requiring_execution_value = ll.mask_as_int(lanes_requiring_execution);
        llvm::Value *execution_required = ll.op_ne(lanes_requiring_execution_value, ll.constant(0));
        then_block = ll.new_basic_block (std::string("then layer ").append(std::to_string(layer)));
        after_block = ll.new_basic_block (std::string("after layer ").append(std::to_string(layer)));
        ll.op_branch (execution_required, then_block, after_block);
        // insert point is now then_block
    } else {
    	lanes_requiring_execution_value = ll.mask_as_int(ll.shader_mask());
    }

    args[2] = lanes_requiring_execution_value;

    std::string name = Strutil::format ("wide_%s_%d", parent->layername().c_str(),
                                        parent->id());
    // Mark the call as a fast call
    llvm::Value *funccall = ll.call_function (name.c_str(), args, 3);
    if (!parent->entry_layer())
        ll.mark_fast_func_call (funccall);

    if (! unconditional)
        ll.op_branch (after_block);  // also moves insert point
}



void
BackendLLVMWide::llvm_run_connected_layers (Symbol &sym, int symindex,
                                             int opnum,
                                             std::set<int> *already_run)
{
    if (sym.valuesource() != Symbol::ConnectedVal)
        return;  // Nothing to do

    OSL_DEV_ONLY(std::cout << "BackendLLVMWide::llvm_run_connected_layers " << sym.name().c_str() << " opnum " << opnum << std::endl);
    bool inmain = (opnum >= inst()->maincodebegin() &&
                   opnum < inst()->maincodeend());

    for (int c = 0;  c < inst()->nconnections();  ++c) {
        const Connection &con (inst()->connection (c));
        // If the connection gives a value to this param
        if (con.dst.param == symindex) {
            // already_run is a set of layers run for this particular op.
            // Just so we don't stupidly do several consecutive checks on
            // whether we ran this same layer. It's JUST for this op.
            if (already_run) {
                if (already_run->count (con.srclayer))
                    continue;  // already ran that one on this op
                else
                    already_run->insert (con.srclayer);  // mark it
            }

            if (inmain) {
                // There is an instance-wide m_layers_already_run that tries
                // to remember which earlier layers have unconditionally
                // been run at any point in the execution of this layer. But
                // only honor (and modify) that when in the main code
                // section, not when in init ops, which are inherently
                // conditional.
                if (m_layers_already_run.count (con.srclayer)) {
                    continue;  // already unconditionally ran the layer
                }
                if (! m_in_conditional[opnum]) {
                    // Unconditionally running -- mark so we don't do it
                    // again. If we're inside a conditional, don't mark
                    // because it may not execute the conditional body.
                    m_layers_already_run.insert (con.srclayer);
                }
            }

            // If the earlier layer it comes from has not yet been
            // executed, do so now.
            llvm_call_layer (con.srclayer);
        }
    }
}



LLVMGEN (llvm_gen_nop)
{
    return true;
}



LLVMGEN (llvm_gen_useparam)
{
    ASSERT (! rop.inst()->unused() &&
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



// Used for printf, error, warning, format
LLVMGEN (llvm_gen_printf)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    // Prepare the args for the call

    // Which argument is the format string?  Usually 0, but for op
    // format(), the formatting string is argument #1.
    int format_arg = (op.opname() == "format" ? 1 : 0);
    Symbol& format_sym = *rop.opargsym (op, format_arg);

    ASSERT(rop.isSymbolUniform(format_sym));

    // For WIDE parameters we want to test the lane first to see
    // if we need to extract values or not
    struct DelayedExtraction
    {
        int argument_slot;
        bool is_float;
        llvm::Value* loaded_value;
    };

    std::vector<DelayedExtraction> delay_extraction_args;
    std::vector<std::vector<llvm::Value*>> call_args;
    if (!format_sym.is_constant()) {
        rop.shadingcontext()->warning ("%s must currently have constant format\n",
                                  op.opname().c_str());
        return false;
    }

    ustring format_ustring = *((ustring*)format_sym.data());
    const char* format = format_ustring.c_str();
    std::string s;
    int arg = format_arg + 1;

    // Check all arguments to see if we will need to generate
    // a seperate printf call for each data lane or not
    // consider the op to be uniform until we find an argument that isn't
    bool op_is_uniform = true;
    for (int a=arg; a < op.nargs(); ++a)
    {
        Symbol& sym (*rop.opargsym (op, a));
        bool arg_is_uniform = rop.isSymbolUniform(sym);
        if (arg_is_uniform == false)
        {
            op_is_uniform = false;
        }
    }
    if (op_is_uniform) {
        call_args.resize(1);
    } else {
        call_args.resize(SimdLaneCount);
    }

    // For some ops, we push the shader globals pointer
    if (op.opname() == op_printf || op.opname() == op_error ||
            op.opname() == op_warning) {
        auto sg = rop.sg_void_ptr();
        llvm::Value * mask = rop.ll.current_mask();
        if (op_is_uniform) {
            call_args[0].push_back (sg);
            call_args[0].push_back (rop.ll.mask_as_int(mask));
        } else {
            // Need to populate all lane's call arguments with same value
            for(int lane_index=0; lane_index < SimdLaneCount; ++lane_index) {
                call_args[lane_index].push_back (sg);
                Mask laneMask(false);
                laneMask.set_on(lane_index);
                call_args[lane_index].push_back (rop.ll.constant(static_cast<int>(laneMask.value())));
            }
        }
    }

    // For some ops, we push the output symbol & mask
    if ((op.opname() == op_format) && (false == op_is_uniform)) {
        Symbol outSymbol = *rop.opargsym (op, 0);

        llvm::Value * outPtr = rop.llvm_void_ptr(outSymbol);
        // Need to populate all lane's call arguments with same value
        for(int lane_index=0; lane_index < SimdLaneCount; ++lane_index) {
            call_args[lane_index].push_back (outPtr);
            Mask laneMask(false);
            laneMask.set_on(lane_index);
            call_args[lane_index].push_back (rop.ll.constant(static_cast<int>(laneMask.value())));
        }
    }

    // We're going to need to adjust the format string as we go, but I'd
    // like to reserve a spot for the char*.
    size_t new_format_slot = call_args[0].size();
    if (op_is_uniform) {
        call_args[0].push_back (NULL);
    } else {
        // Need to populate all lane's call arguments with same value
        for(int lane_index=0; lane_index < SimdLaneCount; ++lane_index) {
            call_args[lane_index].push_back (NULL);
        }
    }

    while (*format != '\0') {
        if (*format == '%') {
            if (format[1] == '%') {
                // '%%' is a literal '%'
                s += "%%";
                format += 2;  // skip both percentages
                continue;
            }
            const char *oldfmt = format;  // mark beginning of format
            while (*format &&
                   *format != 'c' && *format != 'd' && *format != 'e' &&
                   *format != 'f' && *format != 'g' && *format != 'i' &&
                   *format != 'm' && *format != 'n' && *format != 'o' &&
                   *format != 'p' && *format != 's' && *format != 'u' &&
                   *format != 'v' && *format != 'x' && *format != 'X')
                ++format;
            char formatchar = *format++;  // Also eat the format char
            if (arg >= op.nargs()) {
                rop.shadingcontext()->error ("Mismatch between format string and arguments (%s:%d)",
                                        op.sourcefile().c_str(), op.sourceline());
                return false;
            }

            std::string ourformat (oldfmt, format);  // straddle the format
            // Doctor it to fix mismatches between format and data
            Symbol& sym (*rop.opargsym (op, arg));

            bool arg_is_uniform = rop.isSymbolUniform(sym);

            TypeDesc simpletype (sym.typespec().simpletype());
            int num_elements = simpletype.numelements();
            int num_components = simpletype.aggregate;
            if ((sym.typespec().is_closure_based() ||
                 simpletype.basetype == TypeDesc::STRING)
                && formatchar != 's') {
                ourformat[ourformat.length()-1] = 's';
            }
            if (simpletype.basetype == TypeDesc::INT && formatchar != 'd' &&
                formatchar != 'i' && formatchar != 'o' && formatchar != 'u' &&
                formatchar != 'x' && formatchar != 'X') {
                ourformat[ourformat.length()-1] = 'd';
            }
            if (simpletype.basetype == TypeDesc::FLOAT && formatchar != 'f' &&
                formatchar != 'g' && formatchar != 'c' && formatchar != 'e' &&
                formatchar != 'm' && formatchar != 'n' && formatchar != 'p' &&
                formatchar != 'v') {
                ourformat[ourformat.length()-1] = 'f';
            }
            // NOTE(boulos): Only for debug mode do the derivatives get printed...
            for (int a = 0;  a < num_elements;  ++a) {
                llvm::Value *arrind = simpletype.arraylen ? rop.ll.constant(a) : NULL;
                if (sym.typespec().is_closure_based()) {
                    s += ourformat;
                    llvm::Value *v = rop.llvm_load_value (sym, 0, arrind, 0);
                	ASSERT(0 && "incomplete");
                    v = rop.ll.call_function ("osl_closure_to_string", rop.sg_void_ptr(), v);
                    call_args[0].push_back (v);
                    continue;
                }

                for (int c = 0; c < num_components; c++) {
                    if (c != 0 || a != 0)
                        s += " ";
                    s += ourformat;

                    llvm::Value* loaded = rop.llvm_load_value (sym, 0, arrind, c);

                    if (arg_is_uniform) {
                        if (simpletype.basetype == TypeDesc::FLOAT) {
                            // C varargs convention upconverts float->double.
                            loaded = rop.ll.op_float_to_double(loaded);
                        }
                        if (op_is_uniform) {
                            call_args[0].push_back (loaded);
                        } else {
                            // Need to populate all lane's call arguments with same value
                            for(int lane_index=0; lane_index < SimdLaneCount; ++lane_index) {
                                call_args[lane_index].push_back (loaded);
                            }
                        }
                    } else {
                        ASSERT(false == op_is_uniform);
                        delay_extraction_args.push_back(DelayedExtraction{static_cast<int>(call_args[0].size()), simpletype.basetype == TypeDesc::FLOAT, loaded});
                        // Need to populate all lane's call arguments with a place holder
                        // that we can fill in later once we test the lane
                        for(int lane_index=0; lane_index < SimdLaneCount; ++lane_index) {
                            call_args[lane_index].push_back (nullptr);
                        }

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
        std::string prefix = Strutil::format ("Shader %s [%s]: ",
                                              op.opname().c_str(),
                                              rop.inst()->shadername().c_str());
        s = prefix + s;
    }

    // Now go back and put the new format string in its place
    auto llvm_new_format_string = rop.ll.constant (s.c_str());
    if (op_is_uniform) {
        call_args[0][new_format_slot] = llvm_new_format_string;
    } else {
        // Need to populate all lane's call arguments with same value
        for(int lane_index=0; lane_index < SimdLaneCount; ++lane_index) {
            call_args[lane_index][new_format_slot] = llvm_new_format_string;
        }
    }

    // Construct the function name and call it.
    std::string opname = std::string("osl_") + op.opname().string();
    if ((op.opname() != op_format) || (false == op_is_uniform)) {
        opname += std::string("_batched");
    }
    if (op_is_uniform) {
        llvm::Value *ret = rop.ll.call_function (opname.c_str(), &call_args[0][0],
                                                   (int)call_args[0].size());

        // The format op returns a string value, put in in the right spot
        if (op.opname() == op_format)
            rop.llvm_store_value (ret, *rop.opargsym (op, 0));
    } else {

    	// Could be printing wide value at top scope (no mask)
    	// so no need to add a conditional to check
        llvm::Value *mask = rop.ll.current_mask();

        for(int lane_index=0; lane_index < SimdLaneCount; ++lane_index) {

            llvm::BasicBlock* after_block = nullptr;
            if (mask)
            {
                llvm::Value *lane_is_active =  rop.ll.test_mask_lane(mask, lane_index);

                // skip the printf if the lane is not active
                llvm::BasicBlock* then_block = rop.ll.new_basic_block ("test_lane_then");
                after_block = rop.ll.new_basic_block ("test_lane_after");
                rop.ll.op_branch (lane_is_active, then_block, after_block);

            }
            for(const DelayedExtraction &de : delay_extraction_args)
            {
                llvm::Value* scalar_val = rop.ll.op_extract(de.loaded_value, lane_index);

                if (de.is_float) {
                    // C varargs convention upconverts float->double.
                    scalar_val = rop.ll.op_float_to_double(scalar_val);
                }
                call_args[lane_index][de.argument_slot] = scalar_val;

            }
            rop.ll.call_function (opname.c_str(), &call_args[lane_index][0],
                                                       (int)call_args[lane_index].size());

            if (after_block) {
                rop.ll.op_branch (after_block);  // insert point is now after_block
            }
        }
    }

    return true;
}



LLVMGEN (llvm_gen_add)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    bool op_is_uniform = rop.isSymbolUniform(A) && rop.isSymbolUniform(B);
    bool result_is_uniform = rop.isSymbolUniform(Result);
	ASSERT(op_is_uniform || !result_is_uniform);

    ASSERT (! A.typespec().is_array() && ! B.typespec().is_array());
    if (Result.typespec().is_closure()) {
    	ASSERT(0 && "incomplete");
        ASSERT (A.typespec().is_closure() && B.typespec().is_closure());
        llvm::Value *valargs[3];
        valargs[0] = rop.sg_void_ptr();
        valargs[1] = rop.llvm_load_value (A);
        valargs[2] = rop.llvm_load_value (B);
    	ASSERT(0 && "incomplete");
        llvm::Value *res = rop.ll.call_function ("osl_add_closure_closure", valargs, 3);
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

    bool op_is_uniform = rop.isSymbolUniform(A) && rop.isSymbolUniform(B);
    bool result_is_uniform = rop.isSymbolUniform(Result);
	ASSERT(op_is_uniform || !result_is_uniform);

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;

    ASSERT (! Result.typespec().is_closure_based() &&
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

    bool op_is_uniform = rop.isSymbolUniform(A) && rop.isSymbolUniform(B);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = !Result.typespec().is_closure_based() && Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    bool resultIsUniform = rop.isSymbolUniform(Result);
    ASSERT(op_is_uniform || !resultIsUniform);

    // multiplication involving closures
    if (Result.typespec().is_closure()) {
    	ASSERT(0 && "incomplete");
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
    	ASSERT(0 && "incomplete");
        llvm::Value *res = tfloat ? rop.ll.call_function ("osl_mul_closure_float", valargs, 3)
                                  : rop.ll.call_function ("osl_mul_closure_color", valargs, 3);
        rop.llvm_store_value (res, Result, 0, NULL, 0);
        return true;
    }

    // multiplication involving matrices
    if (Result.typespec().is_matrix()) {
    	std::string func_name("osl_mul_");
    	func_name.append(arg_typecode(Result,false,op_is_uniform));
    	func_name.append(arg_typecode(A,false,op_is_uniform));
    	func_name.append(arg_typecode(B,false,op_is_uniform));
        rop.llvm_call_function (func_name.c_str(), Result, A, B, false /*deriv_ptrs*/, op_is_uniform, false /*functionIsLlvmInlined*/,  true /*ptrToReturnStructIs1stArg*/);

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
            ASSERT (is_float);
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

    bool op_is_uniform = rop.isSymbolUniform(A) && rop.isSymbolUniform(B);
    bool resultIsUniform = rop.isSymbolUniform(Result);
    ASSERT(op_is_uniform || !resultIsUniform);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    ASSERT (! Result.typespec().is_closure_based());

    // division involving matrices
    if (Result.typespec().is_matrix()) {
    	std::string func_name("osl_div_");
    	func_name.append(arg_typecode(Result,false,op_is_uniform));
    	func_name.append(arg_typecode(A,false,op_is_uniform));
    	func_name.append(arg_typecode(B,false,op_is_uniform));
    	{
            LLVM_Util::ScopedMasking require_mask_be_passed;
            if (!op_is_uniform && B.typespec().is_matrix()) {
                // We choose to only support masked version of these functions:
                // osl_div_w16mw16fw16m
                // osl_div_w16mw16mw16m
                ASSERT(A.typespec().is_matrix() || A.typespec().is_float());
                ASSERT(Result.typespec().is_matrix() && !resultIsUniform);
                // Because then check the matrices to see if they are affine
                // and take a slow path if not.  Unmasked lanes wold most
                // likely take the slow path, which could have been avoided
                // if we passed the mask in.
                require_mask_be_passed = rop.ll.create_masking_scope(/*enabled=*/true);
            }
            rop.llvm_call_function (func_name.c_str(), Result, A, B, false /*deriv_ptrs*/, op_is_uniform, false /*functionIsLlvmInlined*/,  true /*ptrToReturnStructIs1stArg*/);
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


    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.llvm_load_value (A, 0, i, type, op_is_uniform);
        llvm::Value *b = rop.llvm_load_value (B, 0, i, type, op_is_uniform);
        if (!a || !b)
            return false;

        llvm::Value * b_not_zero;
        llvm::Value *a_div_b;
        if (B.is_constant() && ! rop.is_zero(B)) {
            a_div_b = rop.ll.op_div (a, b);
        } else {
        	// safe_div, implement here vs. calling a function
            b_not_zero = rop.ll.op_ne(b, c_zero);
            if (is_float) {
            	a_div_b = rop.ll.op_select (b_not_zero, rop.ll.op_div (a, b), c_zero);
            } else {
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
            ASSERT (is_float);
            llvm::Value *binv;
            if (B.is_constant() && ! rop.is_zero(B)) {
				binv = rop.ll.op_div (c_one, b);
            } else {
				binv = rop.ll.op_select (b_not_zero, rop.ll.op_div (c_one, b), c_zero);
            }
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
    bool is_float = Result.typespec().is_floatbased();

    bool op_is_uniform = rop.isSymbolUniform(A) && rop.isSymbolUniform(B);
    bool result_is_uniform = rop.isSymbolUniform(Result);
	ASSERT(op_is_uniform || !result_is_uniform);

    int num_components = type.aggregate;
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

    if (Result.has_derivs()) {
        ASSERT (is_float);
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



LLVMGEN (llvm_gen_neg)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);

    bool op_is_uniform = rop.isSymbolUniform(A);
    bool result_is_uniform = rop.isSymbolUniform(Result);
	ASSERT(op_is_uniform || !result_is_uniform);

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int d = 0;  d < 3;  ++d) {  // dx, dy
        for (int i = 0; i < num_components; i++) {
            llvm::Value *a = rop.llvm_load_value (A, d, i, type, op_is_uniform);
            llvm::Value *r = rop.ll.op_neg (a);
        	if (op_is_uniform && !result_is_uniform)
        	{
        		r = rop.ll.widen_value(r);
        	}
            rop.llvm_store_value (r, Result, d, i);
        }
        if (! Result.has_derivs())
            break;
    }
    return true;
}



// Implementation for clamp
LLVMGEN (llvm_gen_clamp)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& X = *rop.opargsym (op, 1);
    Symbol& Min = *rop.opargsym (op, 2);
    Symbol& Max = *rop.opargsym (op, 3);

    bool op_is_uniform = rop.isSymbolUniform(X) && rop.isSymbolUniform(Min) && rop.isSymbolUniform(Max);
    bool result_is_uniform = rop.isSymbolUniform(Result);
	ASSERT(op_is_uniform || !result_is_uniform);

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value *val = rop.llvm_load_value (X, 0, i, type, op_is_uniform);
        llvm::Value *min = rop.llvm_load_value (Min, 0, i, type, op_is_uniform);
        llvm::Value *cond = rop.ll.op_lt (val, min);
        val = rop.ll.op_select (cond, min, val);
        llvm::Value *valdx=NULL, *valdy=NULL;
        if (Result.has_derivs()) {
            valdx = rop.llvm_load_value (X, 1, i, type, op_is_uniform);
            valdy = rop.llvm_load_value (X, 2, i, type, op_is_uniform);
            llvm::Value *mindx = rop.llvm_load_value (Min, 1, i, type, op_is_uniform);
            llvm::Value *mindy = rop.llvm_load_value (Min, 2, i, type, op_is_uniform);
            valdx = rop.ll.op_select (cond, mindx, valdx);
            valdy = rop.ll.op_select (cond, mindy, valdy);
        }
        // Now do the upper bound
        llvm::Value *max = rop.llvm_load_value (Max, 0, i, type, op_is_uniform);
        cond = rop.ll.op_gt (val, max);
        val = rop.ll.op_select (cond, max, val);
        if (Result.has_derivs()) {
            llvm::Value *maxdx = rop.llvm_load_value (Max, 1, i, type, op_is_uniform);
            llvm::Value *maxdy = rop.llvm_load_value (Max, 2, i, type, op_is_uniform);
            valdx = rop.ll.op_select (cond, maxdx, valdx);
            valdy = rop.ll.op_select (cond, maxdy, valdy);
        }

    	if (op_is_uniform && !result_is_uniform)
    	{
    		val = rop.ll.widen_value(val);
    		valdx = rop.ll.widen_value(valdx);
    		valdy = rop.ll.widen_value(valdy);
    	}

        rop.llvm_store_value (val, Result, 0, i);
        rop.llvm_store_value (valdx, Result, 1, i);
        rop.llvm_store_value (valdy, Result, 2, i);
    }
    return true;
}



LLVMGEN (llvm_gen_mix)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);
    Symbol& X = *rop.opargsym (op, 3);

    bool op_is_uniform = rop.isSymbolUniform(Result);

    TypeDesc type = Result.typespec().simpletype();
    ASSERT (!Result.typespec().is_closure_based() &&
            Result.typespec().is_floatbased());
    int num_components = type.aggregate;
    int x_components = X.typespec().aggregate();
    bool derivs = (Result.has_derivs() &&
                   (A.has_derivs() || B.has_derivs() || X.has_derivs()));

    llvm::Value *one;
    if (op_is_uniform)
        one = rop.ll.constant (1.0f);
    else
        one = rop.ll.wide_constant (1.0f);

    llvm::Value *x = rop.llvm_load_value (X, 0, 0, type, op_is_uniform);
    llvm::Value *one_minus_x = rop.ll.op_sub (one, x);
    llvm::Value *xx = derivs ? rop.llvm_load_value (X, 1, 0, type, op_is_uniform) : NULL;
    llvm::Value *xy = derivs ? rop.llvm_load_value (X, 2, 0, type, op_is_uniform) : NULL;
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.llvm_load_value (A, 0, i, type, op_is_uniform);
        llvm::Value *b = rop.llvm_load_value (B, 0, i, type, op_is_uniform);
        if (!a || !b)
            return false;
        if (i > 0 && x_components > 1) {
            // Only need to recompute x and 1-x if they change
            x = rop.llvm_load_value (X, 0, i, type, op_is_uniform);
            one_minus_x = rop.ll.op_sub (one, x);
        }
        // r = a*one_minus_x + b*x
        llvm::Value *r1 = rop.ll.op_mul (a, one_minus_x);
        llvm::Value *r2 = rop.ll.op_mul (b, x);
        llvm::Value *r = rop.ll.op_add (r1, r2);
        rop.llvm_store_value (r, Result, 0, i);

        if (derivs) {
            // mix of duals:
            //   (a*one_minus_x + b*x,
            //    a*one_minus_x.dx + a.dx*one_minus_x + b*x.dx + b.dx*x,
            //    a*one_minus_x.dy + a.dy*one_minus_x + b*x.dy + b.dy*x)
            // and since one_minus_x.dx = -x.dx, one_minus_x.dy = -x.dy,
            //   (a*one_minus_x + b*x,
            //    -a*x.dx + a.dx*one_minus_x + b*x.dx + b.dx*x,
            //    -a*x.dy + a.dy*one_minus_x + b*x.dy + b.dy*x)
            llvm::Value *ax = rop.llvm_load_value (A, 1, i, type, op_is_uniform);
            llvm::Value *bx = rop.llvm_load_value (B, 1, i, type, op_is_uniform);
            if (i > 0 && x_components > 1)
                xx = rop.llvm_load_value (X, 1, i, type, op_is_uniform);
            llvm::Value *rx1 = rop.ll.op_mul (a, xx);
            llvm::Value *rx2 = rop.ll.op_mul (ax, one_minus_x);
            llvm::Value *rx = rop.ll.op_sub (rx2, rx1);
            llvm::Value *rx3 = rop.ll.op_mul (b, xx);
            rx = rop.ll.op_add (rx, rx3);
            llvm::Value *rx4 = rop.ll.op_mul (bx, x);
            rx = rop.ll.op_add (rx, rx4);

            llvm::Value *ay = rop.llvm_load_value (A, 2, i, type, op_is_uniform);
            llvm::Value *by = rop.llvm_load_value (B, 2, i, type, op_is_uniform);
            if (i > 0 && x_components > 1)
                xy = rop.llvm_load_value (X, 2, i, type, op_is_uniform);
            llvm::Value *ry1 = rop.ll.op_mul (a, xy);
            llvm::Value *ry2 = rop.ll.op_mul (ay, one_minus_x);
            llvm::Value *ry = rop.ll.op_sub (ry2, ry1);
            llvm::Value *ry3 = rop.ll.op_mul (b, xy);
            ry = rop.ll.op_add (ry, ry3);
            llvm::Value *ry4 = rop.ll.op_mul (by, x);
            ry = rop.ll.op_add (ry, ry4);

            rop.llvm_store_value (rx, Result, 1, i);
            rop.llvm_store_value (ry, Result, 2, i);
        }
    }

    if (Result.has_derivs() && !derivs) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs (Result);
    }

    return true;
}


LLVMGEN (llvm_gen_select)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);
    Symbol& X = *rop.opargsym (op, 3);
    TypeDesc type = Result.typespec().simpletype();
    ASSERT (!Result.typespec().is_closure_based() &&
            Result.typespec().is_floatbased());
    int num_components = type.aggregate;
    int x_components = X.typespec().aggregate();
    bool derivs = (Result.has_derivs() &&
                   (A.has_derivs() || B.has_derivs()));

    llvm::Value *zero = X.typespec().is_int() ? rop.ll.constant (0)
                                              : rop.ll.constant (0.0f);
    llvm::Value *cond[3];
    for (int i = 0; i < x_components; ++i)
        cond[i] = rop.ll.op_ne (rop.llvm_load_value (X, 0, i), zero);

    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.llvm_load_value (A, 0, i, type);
        llvm::Value *b = rop.llvm_load_value (B, 0, i, type);
        llvm::Value *c = (i >= x_components) ? cond[0] : cond[i];
        llvm::Value *r = rop.ll.op_select (c, b, a);
        rop.llvm_store_value (r, Result, 0, i);
        if (derivs) {
            for (int d = 1; d < 3; ++d) {
                a = rop.llvm_load_value (A, d, i, type);
                b = rop.llvm_load_value (B, d, i, type);
                r = rop.ll.op_select (c, b, a);
                rop.llvm_store_value (r, Result, d, i);
            }
        }
    }

    if (Result.has_derivs() && !derivs) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs (Result);
    }
    return true;
}




// Implementation for min/max
LLVMGEN (llvm_gen_minmax)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& x = *rop.opargsym (op, 1);
    Symbol& y = *rop.opargsym (op, 2);

    bool op_is_uniform = rop.isSymbolUniform(x) && rop.isSymbolUniform(y);
    bool result_is_uniform = rop.isSymbolUniform(Result);

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value *x_val = rop.llvm_load_value (x, 0, i, type, op_is_uniform);
        llvm::Value *y_val = rop.llvm_load_value (y, 0, i, type, op_is_uniform);

        llvm::Value* cond = NULL;
        // NOTE(boulos): Using <= instead of < to match old behavior
        // (only matters for derivs)
        if (op.opname() == op_min) {
            cond = rop.ll.op_le (x_val, y_val);
        } else {
            cond = rop.ll.op_gt (x_val, y_val);
        }

        llvm::Value* res_val = rop.ll.op_select (cond, x_val, y_val);
        if (op_is_uniform && !result_is_uniform)
        {
            res_val = rop.ll.widen_value(res_val);
        }
        rop.llvm_store_value (res_val, Result, 0, i);
        if (Result.has_derivs()) {
          llvm::Value* x_dx = rop.llvm_load_value (x, 1, i, type, op_is_uniform);
          llvm::Value* x_dy = rop.llvm_load_value (x, 2, i, type, op_is_uniform);
          llvm::Value* y_dx = rop.llvm_load_value (y, 1, i, type, op_is_uniform);
          llvm::Value* y_dy = rop.llvm_load_value (y, 2, i, type, op_is_uniform);

          llvm::Value* res_dx = rop.ll.op_select(cond, x_dx, y_dx);
          llvm::Value* res_dy = rop.ll.op_select(cond, x_dy, y_dy);
          if (op_is_uniform && !result_is_uniform)
          {
              res_dx = rop.ll.widen_value(res_dx);
              res_dy = rop.ll.widen_value(res_dy);
          }

          rop.llvm_store_value (res_dx, Result, 1, i);
          rop.llvm_store_value (res_dy, Result, 2, i);
        }
    }
    return true;
}



LLVMGEN (llvm_gen_bitwise_binary_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);
    ASSERT (Result.typespec().is_int() && A.typespec().is_int() &&
            B.typespec().is_int());

    llvm::Value *a = rop.loadLLVMValue (A);
    llvm::Value *b = rop.loadLLVMValue (B);
    if (!a || !b)
        return false;
    llvm::Value *r = NULL;
    if (op.opname() == op_bitand)
        r = rop.ll.op_and (a, b);
    else if (op.opname() == op_bitor)
        r = rop.ll.op_or (a, b);
    else if (op.opname() == op_xor)
        r = rop.ll.op_xor (a, b);
    else if (op.opname() == op_shl)
        r = rop.ll.op_shl (a, b);
    else if (op.opname() == op_shr)
        r = rop.ll.op_shr (a, b);
    else
        return false;
    rop.storeLLVMValue (r, Result);
    return true;
}



// Simple (pointwise) unary ops (Abs, ...,
LLVMGEN (llvm_gen_unary_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& dst  = *rop.opargsym (op, 0);
    Symbol& src = *rop.opargsym (op, 1);
    bool dst_derivs = dst.has_derivs();
    int num_components = dst.typespec().simpletype().aggregate;

    bool dst_float = dst.typespec().is_floatbased();
    bool src_float = src.typespec().is_floatbased();

    for (int i = 0; i < num_components; i++) {
        // Get src1/2 component i
        llvm::Value* src_load = rop.loadLLVMValue (src, i, 0);
        if (!src_load) return false;

        llvm::Value* src_val = src_load;

        // Perform the op
        llvm::Value* result = 0;
        ustring opname = op.opname();

        if (opname == op_compl) {
            ASSERT (dst.typespec().is_int());
            result = rop.ll.op_not (src_val);
        } else {
            // Don't know how to handle this.
            rop.shadingcontext()->error ("Don't know how to handle op '%s', eliding the store\n", opname.c_str());
        }

        // Store the result
        if (result) {
            // if our op type doesn't match result, convert
            if (dst_float && !src_float) {
                // Op was int, but we need to store float
                result = rop.ll.op_int_to_float (result);
            } else if (!dst_float && src_float) {
                // Op was float, but we need to store int
                result = rop.ll.op_float_to_int (result);
            } // otherwise just fine
            rop.storeLLVMValue (result, dst, i, 0);
        }

        if (dst_derivs) {
            // mul results in <a * b, a * b_dx + b * a_dx, a * b_dy + b * a_dy>
            rop.shadingcontext()->info ("punting on derivatives for now\n");
            // FIXME!!
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



// Entire array copying
LLVMGEN (llvm_gen_arraycopy)
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


    bool op_is_uniform = rop.isSymbolUniform(Result);

    llvm::Value *c = rop.llvm_load_value(Index);

    if (rop.isSymbolUniform(Index)) {

        if (rop.shadingsys().range_checking()) {
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
               c = rop.ll.call_function ("osl_range_check_batched", args);
               ASSERT (c);
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
    	ASSERT(Index.is_constant() == false);
    	ASSERT(op_is_uniform == false);

        if (rop.shadingsys().range_checking()) {
            // We need a copy of the indices incase the range check clamps them
            llvm::Value * loc_clamped_wide_index = rop.ll.op_alloca(rop.ll.type_wide_int(), 1, std::string("range clamped index:") + Val.name().c_str());
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
            rop.ll.call_function ("osl_range_check_masked", args);
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



// Vector component assignment
LLVMGEN (llvm_gen_compassign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Index = *rop.opargsym (op, 1);
    Symbol& Val = *rop.opargsym (op, 2);

    bool op_is_uniform = rop.isSymbolUniform(Result);

    llvm::Value *c = rop.llvm_load_value(Index);

    if (rop.isSymbolUniform(Index)) {
        if (rop.shadingsys().range_checking()) {
            if (! (Index.is_constant() &&  *(int *)Index.data() >= 0 &&
                   *(int *)Index.data() < 3)) {
                llvm::Value *args[] = { c, rop.ll.constant(3),
                                        rop.ll.constant(Result.name()),
                                        rop.sg_void_ptr(),
                                        rop.ll.constant(op.sourcefile()),
                                        rop.ll.constant(op.sourceline()),
                                        rop.ll.constant(rop.group().name()),
                                        rop.ll.constant(rop.layer()),
                                        rop.ll.constant(rop.inst()->layername()),
                                        rop.ll.constant(rop.inst()->shadername()) };
                c = rop.ll.call_function ("osl_range_check_batched", args);
                ASSERT (c);
            }
        }

		for (int d = 0;  d < 3;  ++d) {  // deriv
			llvm::Value *val = rop.llvm_load_value (Val, d, 0, TypeDesc::TypeFloat, op_is_uniform);
			if (Index.is_constant()) {
				int i = *(int*)Index.data();
				i = Imath::clamp (i, 0, 2);
				rop.llvm_store_value (val, Result, d, i);
			} else {
				rop.llvm_store_component_value (val, Result, d, c);
			}
			if (! Result.has_derivs())  // skip the derivs if we don't need them
				break;
		}
    } else {
    	ASSERT(Index.is_constant() == false);
    	ASSERT(op_is_uniform == false);

        if (rop.shadingsys().range_checking()) {
            // We need a copy of the indices incase the range check clamps them
            llvm::Value * loc_clamped_wide_index = rop.ll.op_alloca(rop.ll.type_wide_int(), 1, std::string("range clamped index:") + Val.name().c_str());
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
               rop.ll.call_function ("osl_range_check_masked", args);
               // Use the range check indices
               // Although as our implementation below doesn't use any
               // out of range values, clamping the indices here
               // is of questionable value
               c = rop.ll.op_load(loc_clamped_wide_index);
       }

    	// As the index is logically bound to 0, 1, or 2
    	// instead of doing a scatter
    	// We can just load all 3 components and blend them based on the index == 0, index == 1, index == 2
    	llvm::Value *comp0Mask = rop.ll.op_eq(c, rop.ll.wide_constant(0));
    	llvm::Value *comp1Mask = rop.ll.op_eq(c, rop.ll.wide_constant(1));
    	llvm::Value *comp2Mask = rop.ll.op_eq(c, rop.ll.wide_constant(2));
    	// If index != 0 && index != 1, assume index == 2
    	// Essentially free clamping

		for (int d = 0;  d < 3;  ++d) {  // deriv

			llvm::Value *val = rop.llvm_load_value (Val, d, 0, TypeDesc::TypeFloat, op_is_uniform);

			llvm::Value *valc0 = rop.llvm_load_value (Result, d, 0, TypeDesc::UNKNOWN, op_is_uniform);
			llvm::Value *valc1 = rop.llvm_load_value (Result, d, 1, TypeDesc::UNKNOWN, op_is_uniform);
			llvm::Value *valc2 = rop.llvm_load_value (Result, d, 2, TypeDesc::UNKNOWN, op_is_uniform);

			llvm::Value *resultc0 = rop.ll.op_select(comp0Mask,val,valc0);
			llvm::Value *resultc1 = rop.ll.op_select(comp1Mask,val,valc1);
			llvm::Value *resultc2 = rop.ll.op_select(comp2Mask,val,valc2);

			rop.llvm_store_value (resultc0, Result, d, 0);
			rop.llvm_store_value (resultc1, Result, d, 1);
			rop.llvm_store_value (resultc2, Result, d, 2);

			if (! Result.has_derivs())  // skip the derivs if we don't need them
				break;
		}
    }
    return true;
}



// Matrix component reference
LLVMGEN (llvm_gen_mxcompref)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& M = *rop.opargsym (op, 1);
    Symbol& Row = *rop.opargsym (op, 2);
    Symbol& Col = *rop.opargsym (op, 3);

    bool op_is_uniform = rop.isSymbolUniform(Result);
    bool components_are_uniform = rop.isSymbolUniform(Row) && rop.isSymbolUniform(Col);

    llvm::Value *row = rop.llvm_load_value (Row, 0, 0, TypeDesc::UNKNOWN, components_are_uniform);
    llvm::Value *col = rop.llvm_load_value (Col, 0, 0, TypeDesc::UNKNOWN, components_are_uniform);

    if (rop.shadingsys().range_checking()) {
        if (components_are_uniform) {
            if (! (Row.is_constant() &&
                   *(int *)Row.data() >= 0 &&
                   *(int *)Row.data() < 4 &&
                    Col.is_constant() &&
                    *(int *)Col.data() >= 0 &&
                    *(int *)Col.data() < 4)) {
                llvm::Value *args[] = { row, rop.ll.constant(4),
                                        rop.ll.constant(M.name()),
                                        rop.sg_void_ptr(),
                                        rop.ll.constant(op.sourcefile()),
                                        rop.ll.constant(op.sourceline()),
                                        rop.ll.constant(rop.group().name()),
                                        rop.ll.constant(rop.layer()),
                                        rop.ll.constant(rop.inst()->layername()),
                                        rop.ll.constant(rop.inst()->shadername()) };
                row = rop.ll.call_function ("osl_range_check_batched", args);
                args[0] = col;
                col = rop.ll.call_function ("osl_range_check_batched", args);
            }
        } else {
            // We need a copy of the indices incase the range check clamps them
            llvm::Value * loc_clamped_wide_index = rop.ll.op_alloca(rop.ll.type_wide_int(), 1, std::string("range clamped row or col:") + M.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(row, loc_clamped_wide_index);
            llvm::Value *args[] = {rop.ll.void_ptr(loc_clamped_wide_index),
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
            rop.ll.call_function ("osl_range_check_masked", args);
            // Use the range check row
            row = rop.ll.op_load(loc_clamped_wide_index);

            // copy the indices into our temporary
            rop.ll.op_unmasked_store(col, loc_clamped_wide_index);
            rop.ll.call_function ("osl_range_check_masked", args);
            // Use the range check col
            col = rop.ll.op_load(loc_clamped_wide_index);
        }
    }

    llvm::Value *val = NULL;
    if (Row.is_constant() && Col.is_constant()) {
        int r = Imath::clamp (((int*)Row.data())[0], 0, 3);
        int c = Imath::clamp (((int*)Col.data())[0], 0, 3);
        int comp = 4 * r + c;
        val = rop.llvm_load_value (M, 0, comp, TypeDesc::TypeFloat, op_is_uniform);
    } else {
        llvm::Value *comp = rop.ll.op_mul (row, components_are_uniform ? rop.ll.constant(4) : rop.ll.wide_constant(4));
        comp = rop.ll.op_add (comp, col);
        val = rop.llvm_load_component_value (M, 0, comp, op_is_uniform, components_are_uniform);
    }
    rop.llvm_store_value (val, Result);
    rop.llvm_zero_derivs (Result);

    return true;
}



// Matrix component assignment
LLVMGEN (llvm_gen_mxcompassign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Row = *rop.opargsym (op, 1);
    Symbol& Col = *rop.opargsym (op, 2);
    Symbol& Val = *rop.opargsym (op, 3);

    bool op_is_uniform = rop.isSymbolUniform(Result);
    bool components_are_uniform = rop.isSymbolUniform(Row) && rop.isSymbolUniform(Col);

    llvm::Value *row = rop.llvm_load_value (Row, 0, 0, TypeDesc::UNKNOWN, components_are_uniform);
    llvm::Value *col = rop.llvm_load_value (Col, 0, 0, TypeDesc::UNKNOWN, components_are_uniform);

    if (rop.shadingsys().range_checking()) {
        if (components_are_uniform) {
            if (! (Row.is_constant() &&
                   *(int *)Row.data() >= 0 &&
                   *(int *)Row.data() < 4 &&
                    Col.is_constant() &&
                    *(int *)Col.data() >= 0 &&
                    *(int *)Col.data() < 4)) {

                llvm::Value *args[] = { row, rop.ll.constant(4),
                                        rop.ll.constant(Result.name()),
                                        rop.sg_void_ptr(),
                                        rop.ll.constant(op.sourcefile()),
                                        rop.ll.constant(op.sourceline()),
                                        rop.ll.constant(rop.group().name()),
                                        rop.ll.constant(rop.layer()),
                                        rop.ll.constant(rop.inst()->layername()),
                                        rop.ll.constant(rop.inst()->shadername()) };
                row = rop.ll.call_function ("osl_range_check_batched", args);

                args[0] = col;
                col = rop.ll.call_function ("osl_range_check_batched", args);
            }
        } else {
            // We need a copy of the indices incase the range check clamps them
            llvm::Value * loc_clamped_wide_index = rop.ll.op_alloca(rop.ll.type_wide_int(), 1, std::string("range clamped row:") + Result.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(row, loc_clamped_wide_index);
            llvm::Value *args[] = { rop.ll.void_ptr(loc_clamped_wide_index),
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
            rop.ll.call_function ("osl_range_check_masked", args);
            // Use the range check row
            row = rop.ll.op_load(loc_clamped_wide_index);

            // copy the indices into our temporary
            rop.ll.op_unmasked_store(col, loc_clamped_wide_index);
            rop.ll.call_function ("osl_range_check_masked", args);
            // Use the range check col
            col = rop.ll.op_load(loc_clamped_wide_index);
        }
    }

    llvm::Value *val = rop.llvm_load_value (Val, 0, 0, TypeDesc::TypeFloat, op_is_uniform);

    if (Row.is_constant() && Col.is_constant()) {
        int r = Imath::clamp (((int*)Row.data())[0], 0, 3);
        int c = Imath::clamp (((int*)Col.data())[0], 0, 3);
        int comp = 4 * r + c;
        rop.llvm_store_value (val, Result, 0, comp);
    } else {
        llvm::Value *comp = rop.ll.op_mul (row, components_are_uniform ? rop.ll.constant(4) : rop.ll.wide_constant(4));
        comp = rop.ll.op_add (comp, col);
        rop.llvm_store_component_value (val, Result, 0, comp, components_are_uniform);
    }
    return true;
}



// Array length
LLVMGEN (llvm_gen_arraylength)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    DASSERT (Result.typespec().is_int() && A.typespec().is_array());

    int len = A.typespec().is_unsized_array() ? A.initializers()
                                              : A.typespec().arraylength();
    rop.llvm_store_value (rop.ll.constant(len), Result);
    return true;
}



// Array reference
LLVMGEN (llvm_gen_aref)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Src = *rop.opargsym (op, 1);
    Symbol& Index = *rop.opargsym (op, 2);

    bool op_is_uniform = rop.isSymbolUniform(Result);
    bool index_is_uniform = rop.isSymbolUniform(Index);

    // Get array index we're interested in
    llvm::Value *index = rop.loadLLVMValue (Index);
    if (! index)
        return false;

    if (rop.shadingsys().range_checking()) {
        if (index_is_uniform) {
            if (! (Index.is_constant() &&  *(int *)Index.data() >= 0 &&
                   *(int *)Index.data() < Src.typespec().arraylength())) {
                llvm::Value *args[] = { index,
                                        rop.ll.constant(Src.typespec().arraylength()),
                                        rop.ll.constant(Src.name()),
                                        rop.sg_void_ptr(),
                                        rop.ll.constant(op.sourcefile()),
                                        rop.ll.constant(op.sourceline()),
                                        rop.ll.constant(rop.group().name()),
                                        rop.ll.constant(rop.layer()),
                                        rop.ll.constant(rop.inst()->layername()),
                                        rop.ll.constant(rop.inst()->shadername()) };
                index = rop.ll.call_function ("osl_range_check_batched", args);
            }
        } else {
            // We need a copy of the indices incase the range check clamps them
            llvm::Value * loc_clamped_wide_index = rop.ll.op_alloca(rop.ll.type_wide_int(), 1, std::string("range clamped index:") + Src.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(index, loc_clamped_wide_index);

            llvm::Value *args[] = { rop.ll.void_ptr(loc_clamped_wide_index),
                                    rop.ll.mask_as_int(rop.ll.current_mask()),
                                    rop.ll.constant(Src.typespec().arraylength()),
                                    rop.ll.constant(Src.name()),
                                    rop.sg_void_ptr(),
                                    rop.ll.constant(op.sourcefile()),
                                    rop.ll.constant(op.sourceline()),
                                    rop.ll.constant(rop.group().name()),
                                    rop.ll.constant(rop.layer()),
                                    rop.ll.constant(rop.inst()->layername()),
                                    rop.ll.constant(rop.inst()->shadername()) };
            rop.ll.call_function ("osl_range_check_masked", args);
            // Use the range check indices
            index = rop.ll.op_load(loc_clamped_wide_index);
        }
    }

    int num_components = Src.typespec().simpletype().aggregate;
    for (int d = 0;  d <= 2;  ++d) {
        for (int c = 0;  c < num_components;  ++c) {
            llvm::Value *val = rop.llvm_load_value (Src, d, index, c, TypeDesc::UNKNOWN, op_is_uniform, index_is_uniform);
            rop.storeLLVMValue (val, Result, c, d);
        }
        if (! Result.has_derivs())
            break;
    }

    return true;
}



// Array assignment
LLVMGEN (llvm_gen_aassign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Index = *rop.opargsym (op, 1);
    Symbol& Src = *rop.opargsym (op, 2);

    bool resultIsUniform = rop.isSymbolUniform(Result);
    bool index_is_uniform = rop.isSymbolUniform(Index);
    ASSERT(index_is_uniform || !resultIsUniform);

    // Get array index we're interested in
    llvm::Value *index = rop.loadLLVMValue (Index);
    if (! index)
        return false;

    if (rop.shadingsys().range_checking()) {
        if (index_is_uniform) {
            if (! (Index.is_constant() &&  *(int *)Index.data() >= 0 &&
                   *(int *)Index.data() < Result.typespec().arraylength())) {
                llvm::Value *args[] = { index,
                                        rop.ll.constant(Result.typespec().arraylength()),
                                        rop.ll.constant(Result.name()),
                                        rop.sg_void_ptr(),
                                        rop.ll.constant(op.sourcefile()),
                                        rop.ll.constant(op.sourceline()),
                                        rop.ll.constant(rop.group().name()),
                                        rop.ll.constant(rop.layer()),
                                        rop.ll.constant(rop.inst()->layername()),
                                        rop.ll.constant(rop.inst()->shadername()) };
                index = rop.ll.call_function ("osl_range_check_batched", args);
            } else {
                // We need a copy of the indices incase the range check clamps them
                llvm::Value * loc_clamped_wide_index = rop.ll.op_alloca(rop.ll.type_wide_int(), 1, std::string("range clamped index:") + Result.name().c_str());
                // copy the indices into our temporary
                rop.ll.op_unmasked_store(index, loc_clamped_wide_index);

                llvm::Value *args[] = { rop.ll.void_ptr(loc_clamped_wide_index),
                                        rop.ll.mask_as_int(rop.ll.current_mask()),
                                        rop.ll.constant(Result.typespec().arraylength()),
                                        rop.ll.constant(Result.name()),
                                        rop.sg_void_ptr(),
                                        rop.ll.constant(op.sourcefile()),
                                        rop.ll.constant(op.sourceline()),
                                        rop.ll.constant(rop.group().name()),
                                        rop.ll.constant(rop.layer()),
                                        rop.ll.constant(rop.inst()->layername()),
                                        rop.ll.constant(rop.inst()->shadername()) };
                rop.ll.call_function ("osl_range_check_masked", args);
                // Use the range check indices
                index = rop.ll.op_load(loc_clamped_wide_index);
            }
        }
    }

    int num_components = Result.typespec().simpletype().aggregate;
    for (int d = 0;  d <= 2;  ++d) {
        for (int c = 0;  c < num_components;  ++c) {
            llvm::Value *val = rop.loadLLVMValue (Src, c, d, TypeDesc::UNKNOWN, resultIsUniform);
            rop.llvm_store_value (val, Result, d, index, c, index_is_uniform);
        }
        if (! Result.has_derivs())
            break;
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
    ASSERT (Result.typespec().is_triple() && X.typespec().is_float() &&
            Y.typespec().is_float() && Z.typespec().is_float() &&
            (using_space == false || Space.typespec().is_string()));

#if 0 && defined(OSL_DEV)
    bool resultIsUniform = rop.isSymbolUniform(Result);
    bool spaceIsUniform = rop.isSymbolUniform(Space);
    bool xIsUniform = rop.isSymbolUniform(X);
    bool yIsUniform = rop.isSymbolUniform(Y);
    bool zIsUniform = rop.isSymbolUniform(Z);
    std::cout << "llvm_gen_construct_color Result=" << Result.name().c_str() << ((resultIsUniform) ? "(uniform)" : "(varying)");
    if (using_space) {
            std::cout << " Space=" << Space.name().c_str() << ((spaceIsUniform) ? "(uniform)" : "(varying)");
    }
    std::cout << " X=" << X.name().c_str() << ((xIsUniform) ? "(uniform)" : "(varying)")
              << " Y=" << Y.name().c_str()<< ((yIsUniform) ? "(uniform)" : "(varying)")
              << " Z=" << Z.name().c_str()<< ((zIsUniform) ? "(uniform)" : "(varying)")
              << std::endl;
#endif
    bool result_is_uniform = rop.isSymbolUniform(Result);

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

        std::string func_name("osl_prepend_color_from_");
        // Ignoring derivs to match existing behavior, see comment below where
        // any derivs on the result are 0'd out
        func_name.append(arg_typecode(Result, false /*derivs*/, result_is_uniform));
        bool space_is_uniform = rop.isSymbolUniform(Space);
        func_name.append(arg_typecode(Space, false /*derivs*/, space_is_uniform));

        llvm::Value *args[4];
        // NOTE:  Shader Globals is only passed to provide access to report an error to the context
        // no implicit dependency on any Shader Globals is necessary
        args[0] = rop.sg_void_ptr ();  // shader globals
        args[1] = rop.llvm_void_ptr (Result, 0);  // color
        args[2] = space_is_uniform ? rop.llvm_load_value (Space) : rop.llvm_void_ptr(Space); // from
        int arg_count = 3;
        if(!result_is_uniform && rop.ll.is_masking_required()) {
            args[arg_count++] = rop.ll.mask_as_int(rop.ll.current_mask());
            func_name.append("_masked");
        } else {
            func_name.append("_batched");
        }

        rop.ll.call_function (func_name.c_str(), args, arg_count);
        // FIXME(deriv): Punt on derivs for color ctrs with space names.
        // We should try to do this right, but we never had it right for
        // the interpreter, to it's probably not an emergency.
        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
    }

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
    ASSERT (Result.typespec().is_triple() && X.typespec().is_float() &&
            Y.typespec().is_float() && Z.typespec().is_float() &&
            (using_space == false || Space.typespec().is_string()));

#if 0 && defined(OSL_DEV)
    bool spaceIsUniform = rop.isSymbolUniform(Space);
    bool xIsUniform = rop.isSymbolUniform(X);
    bool yIsUniform = rop.isSymbolUniform(Y);
    bool zIsUniform = rop.isSymbolUniform(Z);
    std::cout << "llvm_gen_construct_triple Result=" << Result.name().c_str();
    if (using_space) {
            std::cout << " Space=" << Space.name().c_str() << ((spaceIsUniform) ? "(uniform)" : "(varying)");
    }
    std::cout << " X=" << X.name().c_str() << ((xIsUniform) ? "(uniform)" : "(varying)")
              << " Y=" << Y.name().c_str()<< ((yIsUniform) ? "(uniform)" : "(varying)")
              << " Z=" << Z.name().c_str()<< ((zIsUniform) ? "(uniform)" : "(varying)")
              << std::endl;
#endif


    bool space_is_uniform = rop.isSymbolUniform(Space);
    bool op_is_uniform = rop.isSymbolUniform(X) && rop.isSymbolUniform(Y) && rop.isSymbolUniform(Z) && space_is_uniform;

    bool resultIsUniform = rop.isSymbolUniform(Result);
    ASSERT(op_is_uniform || !resultIsUniform);



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

        ASSERT((false == rend->transform_points (NULL, Strings::_emptystring_, Strings::_emptystring_, 0.0f, NULL, NULL, 0, vectype)) && "incomplete");
        // Didn't want to make RenderServices have to deal will all variants of from/to
        // unless it is going to be used, yes it will have to be done though
//        if (rend->transform_points (NULL, from, to, 0.0f, NULL, NULL, 0, vectype)) {
//            // TODO: Handle non-uniform case below minding mask values
//            ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched
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
    		llvm::Value *args[5] = { rop.sg_void_ptr(),
    			rop.ll.void_ptr(transform),
				space_is_uniform ? rop.llvm_load_value(Space) : rop.llvm_void_ptr(Space),
				rop.ll.constant(Strings::common),
    			rop.ll.mask_as_int(rop.ll.current_mask())};
    		// Dynamically build function name
    		std::string func_name;
    		func_name += "osl_build_transform_matrix_";
    		// Ignore derivatives if uneeded or unsupplied
    		func_name += arg_typecode(Space, false, space_is_uniform);
    		func_name += "s"; // to is constant common space
    		func_name += "_masked";

    		succeeded_as_int = rop.ll.call_function (func_name.c_str(), args, std::extent<decltype(args)>::value);
        }
        {
            llvm::Value *args[5] = {
				rop.llvm_void_ptr(Result /* src */),
				rop.llvm_void_ptr(Result /* dest */),
    			rop.ll.void_ptr(transform),
    			succeeded_as_int,
                rop.ll.mask_as_int(rop.ll.current_mask())};

            ASSERT(rop.isSymbolUniform(Result) == false && "unreachable case");
            // definitely not a nonlinear transformation

            // Dynamically build function name
            std::string func_name;
            func_name += "osl_transform_";
            func_name += triple_type.c_str();
            func_name += "_";
            func_name += arg_typecode(Result, Result.has_derivs(), resultIsUniform);
            func_name += arg_typecode(Result, Result.has_derivs(), resultIsUniform);
            func_name.append(warg_lane_count()).append("m"); // transform arg suffix;
            func_name += "_masked";

            rop.ll.call_function (func_name.c_str(), args, std::extent<decltype(args)>::value);
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
LLVMGEN (llvm_gen_matrix)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    int nargs = op.nargs();
    bool using_space = (nargs == 3 || nargs == 18);
    bool using_two_spaces = (nargs == 3 && rop.opargsym(op,2)->typespec().is_string());
    int nfloats = nargs - 1 - (int)using_space;
    ASSERT (nargs == 2 || nargs == 3 || nargs == 17 || nargs == 18);

    bool result_is_uniform = rop.isSymbolUniform(Result);

    if (using_two_spaces) {
    	// Implicit dependencies to shader globals
    	// could mean the result needs to be varying
        llvm::Value *args[5];
        args[0] = rop.sg_void_ptr();  // shader globals
        args[1] = rop.llvm_void_ptr(Result);  // result
        Symbol& From = *rop.opargsym (op, 1);
        Symbol& To = *rop.opargsym (op, 2);
        bool from_is_uniform = rop.isSymbolUniform(From);
        bool to_is_uniform = rop.isSymbolUniform(To);

        args[2] = from_is_uniform ? rop.llvm_load_value(From) : rop.llvm_void_ptr(From);
        args[3] = to_is_uniform ? rop.llvm_load_value(To): rop.llvm_void_ptr(To);
        if (rop.ll.is_masking_required()) {
        	args[4] = rop.ll.mask_as_int(rop.ll.current_mask());
        }

        // Dynamically build width suffix
        std::string func_name("osl_get_from_to_matrix_");
        func_name += arg_typecode(Result, false, result_is_uniform);
        func_name += arg_typecode(From, false, rop.isSymbolUniform(From));
        func_name += arg_typecode(To, false, rop.isSymbolUniform(To));
        func_name += rop.ll.is_masking_required() ? "_masked" : "_batched";

        rop.ll.call_function (func_name.c_str(), args, rop.ll.is_masking_required() ? 5 : 4);
    } else {
        if (nfloats == 1) {
        	llvm::Value *zero;
            if (result_is_uniform)
                zero = rop.ll.constant (0.0f);
            else
            	zero = rop.ll.wide_constant (0.0f);

            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val = ((i%4) == (i/4))
                    ? rop.llvm_load_value (*rop.opargsym(op,1+using_space),0,0,TypeDesc::UNKNOWN,result_is_uniform)
                    : zero;
                rop.llvm_store_value (src_val, Result, 0, i);
            }
        } else if (nfloats == 16) {
            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val = rop.llvm_load_value (*rop.opargsym(op,i+1+using_space),0,0,TypeDesc::UNKNOWN,result_is_uniform);
                rop.llvm_store_value (src_val, Result, 0, i);
            }
        } else {
            ASSERT (0);
        }
        if (using_space) {
        	// Implicit dependencies to shader globals
        	// could mean the result needs to be varying
            llvm::Value *args[4];
            args[0] = rop.sg_void_ptr();  // shader globals
            args[1] = rop.llvm_void_ptr(Result);  // result
            Symbol& From = *rop.opargsym (op, 1);
            bool from_is_uniform = rop.isSymbolUniform(From);
            args[2] = from_is_uniform ? rop.llvm_load_value(From) : rop.llvm_void_ptr(From);
            if (rop.ll.is_masking_required()) {
            	args[3] = rop.ll.mask_as_int(rop.ll.current_mask());
            }

            // Dynamically build width suffix
            std::string func_name("osl_prepend_matrix_from_");
            func_name += arg_typecode(Result, false, result_is_uniform);
            func_name += arg_typecode(From, false, rop.isSymbolUniform(From));
            func_name += rop.ll.is_masking_required() ? "_masked" : "_batched";

            rop.ll.call_function (func_name.c_str(), args, rop.ll.is_masking_required() ? 4 : 3);
        }
    }
    if (Result.has_derivs())
        rop.llvm_zero_derivs (Result);
    return true;
}



/// int getmatrix (fromspace, tospace, M)
LLVMGEN (llvm_gen_getmatrix)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    ASSERT (nargs == 4);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& From = *rop.opargsym (op, 1);
    Symbol& To = *rop.opargsym (op, 2);
    Symbol& M = *rop.opargsym (op, 3);


	// Implicit dependencies to shader globals
	// could mean the result needs to be varying
    bool result_is_uniform = rop.isSymbolUniform(Result);
	ASSERT(rop.isSymbolUniform(M) == result_is_uniform);

    llvm::Value *args[5];
    args[0] = rop.sg_void_ptr();  // shader globals
    args[1] = rop.llvm_void_ptr(M);  // matrix result

    bool from_is_uniform = rop.isSymbolUniform(From);
    bool to_is_uniform = rop.isSymbolUniform(To);

    args[2] = from_is_uniform ? rop.llvm_load_value(From) : rop.llvm_void_ptr(From);
    args[3] = to_is_uniform ? rop.llvm_load_value(To): rop.llvm_void_ptr(To);

    if (rop.ll.is_masking_required()) {
        args[4] = rop.ll.mask_as_int(rop.ll.current_mask());
    }

    // Dynamically build width suffix
    std::string func_name("osl_get_from_to_matrix_");
    func_name += arg_typecode(M, false, result_is_uniform);
    func_name += arg_typecode(From, false, rop.isSymbolUniform(From));
    func_name += arg_typecode(To, false, rop.isSymbolUniform(To));
    func_name += rop.ll.is_masking_required() ? "_masked" : "_batched";

    llvm::Value *result = rop.ll.call_function (func_name.c_str(), args, rop.ll.is_masking_required() ? 5 : 4);
    rop.llvm_conversion_store_masked_status(result, Result);
    rop.llvm_zero_derivs (M);
    return true;
}



// transform{,v,n} (string tospace, triple p)
// transform{,v,n} (string fromspace, string tospace, triple p)
// transform{,v,n} (matrix, triple p)
LLVMGEN (llvm_gen_transform)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    Symbol *Result = rop.opargsym (op, 0);
    Symbol *From = (nargs == 3) ? NULL : rop.opargsym (op, 1);
    Symbol *To = rop.opargsym (op, (nargs == 3) ? 1 : 2);
    Symbol *P = rop.opargsym (op, (nargs == 3) ? 2 : 3);

    bool result_is_uniform = rop.isSymbolUniform(*Result);
    bool to_is_uniform = rop.isSymbolUniform(*To);
    bool P_is_uniform = rop.isSymbolUniform(*P);
    bool from_is_uniform = (From == NULL) ? true : rop.isSymbolUniform(*From);

    TypeDesc::VECSEMANTICS vectype = TypeDesc::POINT;
    // TODO: switch statement with static/extern strings to avoid lookup
    ustring triple_type("point");
    if (op.opname() == "transformv") {
        vectype = TypeDesc::VECTOR;
        triple_type = ustring("vector");
    } else if (op.opname() == "transformn") {
        vectype = TypeDesc::NORMAL;
        triple_type = ustring("normal");
    }

	llvm::Value * transform = nullptr;
	llvm::Value *succeeded_as_int = nullptr;
	std::string transform_arg_suffix;
    if (To->typespec().is_matrix()) {
    	ASSERT(From == NULL);
        // llvm_ops has the matrix version already implemented
        //llvm_gen_generic (rop, opnum);
        //return true;
    	transform_arg_suffix = arg_typecode(*To, false, to_is_uniform);
    	transform = rop.llvm_void_ptr(*To);
    	succeeded_as_int = rop.ll.mask_as_int(rop.ll.current_mask());
    } else {

		// Named space versions from here on out.
		if ((From == NULL || From->is_constant()) && To->is_constant()) {
			// We can know all the space names at this time
			ustring from = From ? *((ustring *)From->data()) : Strings::common;
			ustring to = *((ustring *)To->data());
			ustring syn = rop.shadingsys().commonspace_synonym();
			if (from == syn)
				from = Strings::common;
			if (to == syn)
				to = Strings::common;
			if (from == to) {
				// An identity transformation, just copy
				if (Result != P) // don't bother in-place copy
					rop.llvm_assign_impl (*Result, *P);
				return true;
			}
		}
		//OSL_DEV_ONLY(std::cout << "wide transform 'source space' = " << from << " 'dest space' = " << to << std::endl);

		RendererServices *rend (rop.shadingsys().renderer());

		ASSERT((false == rend->transform_points (NULL, Strings::_emptystring_, Strings::_emptystring_, 0.0f, NULL, NULL, 0, vectype)) && "incomplete");
		// Didn't want to make RenderServices have to deal will all variants of from/to
		// unless it is going to be used, yes it will have to be done though
	//    if (rend->transform_points (NULL, from, to, 0.0f, NULL, NULL, 0, vectype)) {
	//
	//        // TODO: Handle non-uniform case below minding mask values
	//        ASSERT(rop.isSymbolUniform(*Result));
	//        ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched
	//
	//        // renderer potentially knows about a nonlinear transformation.
	//        // Note that for the case of non-constant strings, passing empty
	//        // from & to will make transform_points just tell us if ANY
	//        // nonlinear transformations potentially are supported.
	//        rop.ll.call_function ("osl_transform_triple_nonlinear", args, 8);
	//    } else
		transform = rop.temp_wide_matrix_ptr();
		{
			ASSERT(From != NULL && "expect NULL was replaced by constant folding to a common_space");
			llvm::Value *args[5] = { rop.sg_void_ptr(),
				rop.ll.void_ptr(transform),
				from_is_uniform ? rop.llvm_load_value(*From) : rop.llvm_void_ptr(*From),
				to_is_uniform ? rop.llvm_load_value(*To) : rop.llvm_void_ptr(*To),
				rop.ll.mask_as_int(rop.ll.current_mask())};
			// Dynamically build function name
			std::string func_name;
			func_name += "osl_build_transform_matrix_";
			// Ignore derivatives if uneeded or unsupplied
			func_name += arg_typecode(*From, false, from_is_uniform);
			func_name += arg_typecode(*To, false, to_is_uniform);
			func_name += "_masked";

			succeeded_as_int = rop.ll.call_function (func_name.c_str(), args, std::extent<decltype(args)>::value);
		}
		// The results of looking up a transform are always wide
		transform_arg_suffix.append(warg_lane_count()).append("m");

    }
    {
    	if (result_is_uniform)
    	{
    		ASSERT(to_is_uniform);
    		ASSERT(P_is_uniform);

			llvm::Value *args[] = {
				rop.llvm_void_ptr(*Result),
				rop.ll.void_ptr(transform),
				rop.llvm_void_ptr(*P)};

			// Dynamically build function name
			std::string func_name = std::string("osl_") + op.opname().string() + "_";
			// Ignore derivatives if uneeded or unsupplied
			bool has_derivs = (Result->has_derivs() && P->has_derivs());
			func_name += arg_typecode(*P, has_derivs, P_is_uniform);
			func_name += transform_arg_suffix;
			func_name += arg_typecode(*Result, has_derivs, result_is_uniform);

			rop.ll.call_function (func_name.c_str(), args, std::extent<decltype(args)>::value);
    	} else {
			llvm::Value *args[] = {
				rop.llvm_void_ptr(*P),
				rop.llvm_void_ptr(*Result),
				rop.ll.void_ptr(transform),
				succeeded_as_int,
				rop.ll.mask_as_int(rop.ll.current_mask())};

			// definitely not a nonlinear transformation

			// Dynamically build function name
			std::string func_name;
			func_name += "osl_transform_";
			func_name += triple_type.c_str();
			func_name += "_";
			// Ignore derivatives if uneeded or unsupplied
			bool has_derivs = (Result->has_derivs() && P->has_derivs());
			func_name += arg_typecode(*P, has_derivs, P_is_uniform);
			func_name += arg_typecode(*Result, has_derivs, result_is_uniform);
			func_name += transform_arg_suffix;
			func_name += "_masked";

			rop.ll.call_function (func_name.c_str(), args, std::extent<decltype(args)>::value);
    	}

        // To reduce the number of combinations to support
        // we take on the work of zero'ing out the derivatives here
        // versus adding another version of the functions that just
        // zeros them out.
        // NOTE:  the original scalar version 0's out derivatives
        // regardless of the success of the transformation
        // however the operation mask should still be respected
        if (Result->has_derivs() && !P->has_derivs()) {
        	rop.llvm_zero_derivs (*Result);
        }

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

    for (int i = 0; i < Result.typespec().aggregate(); ++i) {
        llvm::Value* src_val = rop.llvm_load_value (Src, deriv, i);
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

    if (&Src == rop.inst()->symbol(rop.inst()->Psym())) {
        // dPdz -- the only Dz we know how to take
        int deriv = 3;
        for (int i = 0; i < Result.typespec().aggregate(); ++i) {
            llvm::Value* src_val = rop.llvm_load_value (Src, deriv, i);
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



LLVMGEN (llvm_gen_filterwidth)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result (*rop.opargsym (op, 0));
    Symbol& Src (*rop.opargsym (op, 1));

    ASSERT (Src.typespec().is_float() || Src.typespec().is_triple());

    bool op_is_uniform = rop.isSymbolUniform(Result);

    if (Src.has_derivs()) {
        if (op_is_uniform)
        {
			if (Src.typespec().is_float()) {
				// TODO: Handle non-uniform case below minding mask values
				ASSERT(rop.isSymbolUniform(Result));
				llvm::Value *r = rop.ll.call_function ("osl_filterwidth_fdf",
														 rop.llvm_void_ptr (Src));
				rop.llvm_store_value (r, Result);
			} else {
				// TODO: Handle non-uniform case below minding mask values
				ASSERT(rop.isSymbolUniform(Result));

				rop.ll.call_function ("osl_filterwidth_vdv",
										rop.llvm_void_ptr (Result),
										rop.llvm_void_ptr (Src));
			}
			// Don't have 2nd order derivs
			rop.llvm_zero_derivs (Result);
        } else {

            // Dynamically build width suffix
            std::string func_name("osl_filterwidth_");
            // The result may have derivatives, but we zero them out after this
            // function call, so just always treat the result as not having derivates
            func_name += warg_typecode(&Result, false);
            func_name += warg_typecode(&Src, true);

            llvm::Value *args[3];
            args[0] = rop.llvm_void_ptr (Result);
            args[1] = rop.llvm_void_ptr (Src);
            int argCount = 2;

            if (rop.ll.is_masking_required()) {
                func_name += "_masked";
                args[2] = rop.ll.mask_as_int(rop.ll.current_mask());
                argCount = 3;
            }

            rop.ll.call_function (func_name.c_str(),
                                  args, argCount);
			// Don't have 2nd order derivs
			rop.llvm_zero_derivs (Result);
        }
    } else {
        // No derivs to be had
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
    ASSERT (Result.typespec().is_int() && ! Result.has_derivs());

    bool op_is_uniform = rop.isSymbolUniform(A) && rop.isSymbolUniform(B);
    bool result_is_uniform = rop.isSymbolUniform(Result);

    if (A.typespec().is_closure()) {
    	ASSERT(0 && "incomplete");
        ASSERT (B.typespec().is_int() &&
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
    bool float_based = A.typespec().is_floatbased() || B.typespec().is_floatbased();
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
            ASSERT (0 && "Comparison error");
        }
        ASSERT (result);

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
    ASSERT (final_result);

    // Lets not convert comparisons from bool to int

    OSL_DEV_ONLY(std::cout << "About to rop.storeLLVMValue (final_result, Result, 0, 0); op_is_uniform=" << op_is_uniform  << std::endl);
    // Although we try to use llvm bool (i1) for comparison results
    // sometimes we could not force the data type to be an bool and it remains
    // an int, for those cases we will need to convert the boolean to int
	llvm::Type * resultType = rop.ll.llvm_typeof(rop.llvm_get_pointer(Result));
	if ((resultType == reinterpret_cast<llvm::Type *>(rop.ll.type_wide_int_ptr())) ||
		(resultType == reinterpret_cast<llvm::Type *>(rop.ll.type_int_ptr()))) {
		final_result = rop.ll.op_bool_to_int (final_result);
	}

	ASSERT(op_is_uniform || !result_is_uniform);
	if (op_is_uniform && !result_is_uniform)
	{
		final_result = rop.ll.widen_value(final_result);
	}
    rop.storeLLVMValue (final_result, Result, 0, 0);
    OSL_DEV_ONLY(std::cout << "AFTER to rop.storeLLVMValue (final_result, Result, 0, 0);" << std::endl);

    return true;
}



// int regex_search (string subject, string pattern)
// int regex_search (string subject, int results[], string pattern)
// int regex_match (string subject, string pattern)
// int regex_match (string subject, int results[], string pattern)
LLVMGEN (llvm_gen_regex)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    ASSERT (nargs == 3 || nargs == 4);
    Symbol &Result (*rop.opargsym (op, 0));
    Symbol &Subject (*rop.opargsym (op, 1));
    bool do_match_results = (nargs == 4);
    bool fullmatch = (op.opname() == "regex_match");
    Symbol &Match (*rop.opargsym (op, 2));
    Symbol &Pattern (*rop.opargsym (op, 2+do_match_results));
    ASSERT (Result.typespec().is_int() && Subject.typespec().is_string() &&
            Pattern.typespec().is_string());
    ASSERT (!do_match_results ||
            (Match.typespec().is_array() &&
             Match.typespec().elementtype().is_int()));

    std::vector<llvm::Value*> call_args;
    // First arg is ShaderGlobals ptr
    call_args.push_back (rop.sg_void_ptr());
    // Next arg is subject string
    call_args.push_back (rop.llvm_load_value (Subject));
    // Pass the results array and length (just pass 0 if no results wanted).
    call_args.push_back (rop.llvm_void_ptr(Match));
    if (do_match_results)
        call_args.push_back (rop.ll.constant(Match.typespec().arraylength()));
    else
        call_args.push_back (rop.ll.constant(0));
    // Pass the regex match pattern
    call_args.push_back (rop.llvm_load_value (Pattern));
    // Pass whether or not to do the full match
    call_args.push_back (rop.ll.constant(fullmatch));

    // TODO:  probably need to serialize calls to regex, one for reach data lane

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value *ret = rop.ll.call_function ("osl_regex_impl", &call_args[0],
                                               (int)call_args.size());
    rop.llvm_store_value (ret, Result);
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
        if(rop.isSymbolUniform(*s) == false) {
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

    std::string name = std::string("osl_") + op.opname().string() + "_";


    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        if(uniformFormOfFunction == false) {
            // Non uniform, so add the "wide" prefix
            name += "w";
            name += std::to_string(SimdLaneCount);
        }
        if (any_deriv_args && Result.has_derivs() && s->has_derivs() && !s->typespec().is_matrix())
            name += "d";
        if (s->typespec().is_float())
            name += "f";
        else if (s->typespec().is_triple())
            name += "v";
        else if (s->typespec().is_matrix())
            name += "m";
        else if (s->typespec().is_string())
            name += "s";
        else if (s->typespec().is_int())
            name += "i";
        else ASSERT (0);
    }

    OSL_DEV_ONLY(std::cout << "llvm_gen_generic " << name.c_str() << std::endl);

    if (! Result.has_derivs() || ! any_deriv_args) {

        const OpDescriptor *opd = rop.shadingsys().op_descriptor (op.opname());

        bool functionIsLlvmInlined = opd->flags & OpDescriptor::LLVMInlined;

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
        	OSL_DEV_ONLY(std::cout << ">>stores return value " << name.c_str() << std::endl);
            llvm::Value *r = rop.llvm_call_function (name.c_str(),
                                                     &(args[1]), op.nargs()-1,
                                                     /*deriv_ptrs*/ false,
                                                     uniformFormOfFunction,
                                                     functionIsLlvmInlined,
                                                     false /*ptrToReturnStructIs1stArg*/);
            // The store will deal with masking
            rop.llvm_store_value (r, Result);
        } else {
        	OSL_DEV_ONLY(std::cout << ">>return value is pointer " << name.c_str() << std::endl);

            rop.llvm_call_function (name.c_str(),
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
        ASSERT (Result.has_derivs() && any_deriv_args);
        rop.llvm_call_function (name.c_str(),
                                (args.size())? &(args[0]): NULL, op.nargs(),
                                /*deriv_ptrs*/ true, uniformFormOfFunction, false /*functionIsLlvmInlined*/,
                                true /*ptrToReturnStructIs1stArg*/);
    }

    OSL_DEV_ONLY(std::cout << std::endl);

    return true;
}



LLVMGEN (llvm_gen_sincos)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    Symbol& Theta   = *rop.opargsym (op, 0);
    Symbol& Sin_out = *rop.opargsym (op, 1);
    Symbol& Cos_out = *rop.opargsym (op, 2);

    bool theta_deriv   = Theta.has_derivs();
    bool result_derivs = (Sin_out.has_derivs() || Cos_out.has_derivs());

    bool op_is_uniform = rop.isSymbolUniform(Theta);

    ASSERT(op_is_uniform || (!rop.isSymbolUniform(Sin_out) && !rop.isSymbolUniform(Cos_out)));
    // Handle broadcasting results to wide results
    ASSERT((!op_is_uniform || (rop.isSymbolUniform(Sin_out) && rop.isSymbolUniform(Cos_out))) && "incomplete");

    std::string func_name = std::string("osl_sincos_");
    func_name += arg_typecode(Theta, result_derivs  && theta_deriv, op_is_uniform);
    func_name += arg_typecode(Sin_out, Sin_out.has_derivs() && result_derivs  && theta_deriv, op_is_uniform);
    func_name += arg_typecode(Cos_out, Cos_out.has_derivs() && result_derivs  && theta_deriv, op_is_uniform);

    std::vector<llvm::Value *> args;
    if(true ==  ((theta_deriv && result_derivs) || Theta.typespec().is_triple() || !op_is_uniform) ){
      args.push_back(rop.llvm_void_ptr (Theta)) ;
    }
    else {
       args.push_back(rop.llvm_load_value (Theta));
    }
    args.push_back(rop.llvm_void_ptr (Sin_out));
    args.push_back(rop.llvm_void_ptr (Cos_out));

    if (rop.ll.is_masking_required() ) {
        func_name += std::string("_masked");
        args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
    }

    rop.ll.call_function (func_name.c_str(), &args[0], args.size());


    // If the input angle didn't have derivatives, we would not have
    // called the version of sincos with derivs; however in that case we
    // need to clear the derivs of either of the outputs that has them.
    if (Sin_out.has_derivs() && !theta_deriv)
        rop.llvm_zero_derivs (Sin_out);
    if (Cos_out.has_derivs() && !theta_deriv)
        rop.llvm_zero_derivs (Cos_out);

    return true;
}



LLVMGEN (llvm_gen_andor)
{
    Opcode& op (rop.inst()->ops()[opnum]);
    Symbol& result = *rop.opargsym (op, 0);
    Symbol& a = *rop.opargsym (op, 1);
    Symbol& b = *rop.opargsym (op, 2);

    llvm::Value* i1_res = NULL;
    llvm::Value* a_val = rop.llvm_load_value (a, 0, 0, TypeDesc::TypeInt);
    llvm::Value* b_val = rop.llvm_load_value (b, 0, 0, TypeDesc::TypeInt);
    if (op.opname() == op_and) {
        // From the old bitcode generated
        // define i32 @osl_and_iii(i32 %a, i32 %b) nounwind readnone ssp {
        //     %1 = icmp ne i32 %b, 0
        //  %not. = icmp ne i32 %a, 0
        //     %2 = and i1 %1, %not.
        //     %3 = zext i1 %2 to i32
        //   ret i32 %3
        llvm::Value* b_ne_0 = rop.ll.op_ne (b_val, rop.ll.constant(0));
        llvm::Value* a_ne_0 = rop.ll.op_ne (a_val, rop.ll.constant(0));
        llvm::Value* both_ne_0 = rop.ll.op_and  (b_ne_0, a_ne_0);
        i1_res = both_ne_0;
    } else {
        // Also from the bitcode
        // %1 = or i32 %b, %a
        // %2 = icmp ne i32 %1, 0
        // %3 = zext i1 %2 to i32
        llvm::Value* or_ab = rop.ll.op_or(a_val, b_val);
        llvm::Value* or_ab_ne_0 = rop.ll.op_ne (or_ab, rop.ll.constant(0));
        i1_res = or_ab_ne_0;
    }

    // Although we try to use llvm bool (i1) for comparison results
    // sometimes we could not force the data type to be an bool and it remains
    // an int, for those cases we will need to convert the boolean to int
	llvm::Type * resultType = rop.ll.llvm_typeof(rop.llvm_get_pointer(result));
	if ((resultType == reinterpret_cast<llvm::Type *>(rop.ll.type_wide_int_ptr())) ||
		(resultType == reinterpret_cast<llvm::Type *>(rop.ll.type_int_ptr()))) {
		llvm::Value* final_result = rop.ll.op_bool_to_int (i1_res);
		// TODO: should llvm_store_value handle this internally,
		// To make sure we don't miss any scenarios
		rop.llvm_store_value(final_result, result, 0, 0);
	} else {
		rop.llvm_store_value(i1_res, result, 0, 0);
	}
    return true;
}




LLVMGEN (llvm_gen_if)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym (op, 0);

    const char * cond_name = cond.name().c_str();
    bool op_is_uniform = rop.isSymbolUniform(cond);

    bool elseBlockRequired = op.jump(0) != op.jump(1);

	int beforeThenElseReturnCount = rop.ll.masked_return_count();
    int beforeThenElseBreakCount = rop.ll.masked_break_count();
    int beforeThenElseContinueCount = rop.ll.masked_continue_count();

    if (op_is_uniform) {
        // Load the condition variable and figure out if it's nonzero
        llvm::Value* cond_val = rop.llvm_test_nonzero (cond);

        // Branch on the condition, to our blocks
        llvm::BasicBlock* then_block = rop.ll.new_basic_block (std::string("then (uniform)") + cond_name);
        llvm::BasicBlock* else_block = elseBlockRequired ?
        		                       rop.ll.new_basic_block (std::string("else (uniform)") + cond_name) :
									   nullptr;
        llvm::BasicBlock* after_block = rop.ll.new_basic_block (std::string("after_if (uniform)") + cond_name);
        rop.ll.op_branch (cond_val, then_block, elseBlockRequired ? else_block : after_block);

        // Then block
        rop.build_llvm_code (opnum+1, op.jump(0), then_block);
        rop.ll.op_branch (after_block); // insert point is now after_block
		if (elseBlockRequired) {
	        // Else block
	        rop.build_llvm_code (op.jump(0), op.jump(1), else_block);
	        rop.ll.op_branch (after_block);  // insert point is now after_block
		}

        // NOTE: if a return or exit is encounter inside a uniform
        // conditional block, then it will branch to the last
        // rop.ll.push_masked_return_block(...)
        // or if there is none, operate in a scalar fashion
        // branching to the return_block() or exit_instance()
    } else {

        llvm::Value* mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
        if (mask->getType() != rop.ll.type_wide_bool()) {
            ASSERT(mask->getType() == rop.ll.type_wide_int());
            mask = rop.ll.op_int_to_bool(mask);
        }
        ASSERT(mask->getType() == rop.ll.type_wide_bool());
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("if",mask);
#endif
        rop.ll.push_mask(mask);

        // TODO:  Add heuristic to control if we can avoid testing
        // for any lanes active and just execute masked.
        // However must make sure the then or else block does not
        // contain a call to a lower level, those must not be executed
        // if the mask is all off

        // We use the combined mask stack + the if condition's mask we aready pushed
		llvm::Value* anyThenLanesActive = rop.ll.test_if_mask_is_non_zero(rop.ll.current_mask());

		// Branch on the condition, to our blocks
		llvm::BasicBlock* then_block = rop.ll.new_basic_block (std::string("then (varying)") + cond_name);

		llvm::BasicBlock* test_else_block = elseBlockRequired ? rop.ll.new_basic_block (std::string("test_else (varying)") + cond_name) : nullptr;
		llvm::BasicBlock* else_block = elseBlockRequired ? rop.ll.new_basic_block (std::string("else (varying)") + cond_name) : nullptr;

		llvm::BasicBlock* after_block = rop.ll.new_basic_block (std::string("after_if (varying)") + cond_name);

		// Then block
		// Perhaps mask should be parameter to build_llvm_code?
		rop.ll.op_branch (anyThenLanesActive, then_block, elseBlockRequired ? test_else_block : after_block);

		rop.ll.set_insert_point (then_block);
		//rop.ll.push_mask(mask); // we pushed this mask before the then block so we can test for 0 active lanes
		rop.ll.push_masked_return_block(elseBlockRequired ? test_else_block : after_block);
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("then");
#endif
		rop.build_llvm_code (opnum+1, op.jump(0), then_block);
		rop.ll.pop_masked_return_block();
		rop.ll.pop_mask();
		// Execute both the "then" and the "else" blocks with masking
		rop.ll.op_branch (elseBlockRequired ? test_else_block : after_block);
		if (elseBlockRequired) {
			// Else block
			// insertion point should be test_else_block
			rop.ll.push_mask(mask, true /* negate */);
            llvm::Value* anyElseLanesActive = rop.ll.test_if_mask_is_non_zero(rop.ll.current_mask());

			rop.ll.op_branch (anyElseLanesActive, else_block, after_block);
			rop.ll.set_insert_point (else_block);
			rop.ll.push_masked_return_block(after_block);
#ifdef __OSL_TRACE_MASKS
            rop.llvm_print_mask("else");
#endif
			rop.build_llvm_code (op.jump(0), op.jump(1), else_block);
			rop.ll.pop_masked_return_block();
			rop.ll.pop_mask();
			rop.ll.op_branch (after_block);
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
		llvm::Value* anyLanesActive = rop.ll.test_if_mask_is_non_zero(rop.ll.current_mask());

		llvm::BasicBlock * nextMaskScope;
		if (rop.ll.has_masked_return_block()) {
			nextMaskScope = rop.ll.masked_return_block();
		} else {
			nextMaskScope = rop.ll.inside_function() ?
							rop.ll.return_block() :
							rop.llvm_exit_instance_block();
		}
		llvm::BasicBlock* after_applying_return_block = rop.ll.new_basic_block (std::string("after_if_applied_return_mask (varying)") + cond_name);

		rop.ll.op_branch (anyLanesActive, after_applying_return_block, nextMaskScope);
	}

    // Continue on with the previous flow
    return true;
}


LLVMGEN (llvm_gen_loop_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym (op, 0);

    bool op_is_uniform = rop.isSymbolUniform(cond);
    const char * cond_name = cond.name().c_str();

    if (op_is_uniform) {
    	OSL_DEV_ONLY(std::cout << "llvm_gen_loop_op UNIFORM based on " << cond.name().c_str() << std::endl);

        // Branch on the condition, to our blocks
        llvm::BasicBlock* cond_block = rop.ll.new_basic_block (std::string("cond (uniform)") + cond_name);
        llvm::BasicBlock* body_block = rop.ll.new_basic_block (std::string("body (uniform)") + cond_name);
        llvm::BasicBlock* step_block = rop.ll.new_basic_block (std::string("step (uniform)") + cond_name);
        llvm::BasicBlock* after_block = rop.ll.new_basic_block (std::string("after_loop (uniform)") + cond_name);
        // Save the step and after block pointers for possible break/continue
        rop.ll.push_loop (step_block, after_block);
        // We need to track uniform loops as well
        // to properly handle a uniform loop inside of a varying loop
        // and since the "break" op has no symbol for us to check for
        // uniformity, it can check the current masked loop condition location
        // to see if it is null or not (uniform vs. varying)
        rop.ll.push_masked_loop(nullptr, nullptr);

        // Initialization (will be empty except for "for" loops)
        rop.build_llvm_code (opnum+1, op.jump(0));

        // For "do-while", we go straight to the body of the loop, but for
        // "for" or "while", we test the condition next.
        rop.ll.op_branch (op.opname() == op_dowhile ? body_block : cond_block);

        // Load the condition variable and figure out if it's nonzero
        rop.build_llvm_code (op.jump(0), op.jump(1), cond_block);
        llvm::Value* cond_val = rop.llvm_test_nonzero (cond);

        // Jump to either LoopBody or AfterLoop
        rop.ll.op_branch (cond_val, body_block, after_block);

        // Body of loop
        rop.build_llvm_code (op.jump(1), op.jump(2), body_block);
        rop.ll.op_branch (step_block);

        // Step
        rop.build_llvm_code (op.jump(2), op.jump(3), step_block);
        rop.ll.op_branch (cond_block);

        // Continue on with the previous flow
        rop.ll.set_insert_point (after_block);
        rop.ll.pop_masked_loop();
        rop.ll.pop_loop ();
    } else {
    	OSL_DEV_ONLY(std::cout << "llvm_gen_loop_op VARYING based on " << cond.name().c_str() << std::endl);

        // Branch on the condition, to our blocks
        llvm::BasicBlock* cond_block;
        llvm::BasicBlock* body_block;
        // Improve readability of generated IR by creating basic blocks in the order they
        // will be processed
        if (op.opname() == op_dowhile) {
        	body_block = rop.ll.new_basic_block (std::string("body (varying):") + cond_name);
        	cond_block = rop.ll.new_basic_block (std::string("cond (varying):") + cond_name);
        } else {
        	cond_block = rop.ll.new_basic_block (std::string("cond (varying):") + cond_name);
        	body_block = rop.ll.new_basic_block (std::string("body (varying):") + cond_name);
        }
        llvm::BasicBlock* step_block = rop.ll.new_basic_block (std::string("step (varying):") + cond_name);
        llvm::BasicBlock* after_block = rop.ll.new_basic_block (std::string("after_loop (varying):") + cond_name);

		int return_count_before_loop = rop.ll.masked_return_count();

        // Save the step and after block pointers for possible break/continue
        rop.ll.push_loop (step_block, after_block);

        bool loopHasContinue = rop.loopHasContinue(opnum);

        llvm::Value * loc_of_continue_mask = loopHasContinue ? rop.ll.op_alloca(rop.ll.type_wide_bool(), 1, std::string("continue mask:") + cond_name) : nullptr;
        rop.ll.push_masked_loop(rop.llvm_get_pointer (cond), loc_of_continue_mask);

        // Initialization (will be empty except for "for" loops)
        rop.build_llvm_code (opnum+1, op.jump(0));

        // Store current top of the mask stack (or all 1's) as the current mask value
        // as we enter the loop
        llvm::Value* initial_mask = rop.ll.current_mask();
        rop.ll.op_unmasked_store(initial_mask, rop.llvm_get_pointer (cond));

    	// If all lanes inside the loop become inactive,
    	// jump to the step as it may have been cause by a continue.
    	// If no continue is possible, then we can just jump to the
    	// after_block when all lanes become inactive
		rop.ll.push_masked_return_block(loopHasContinue ? step_block : after_block);

        // For "do-while", we go straight to the body of the loop, but for
        // "for" or "while", we test the condition next.
        if (op.opname() == op_dowhile) {
            rop.ll.op_branch (body_block);

            llvm::Value* pre_condition_mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
            ASSERT(pre_condition_mask->getType() == rop.ll.type_wide_bool());

            rop.ll.push_mask(pre_condition_mask, false /* negate */, true /* absolute */);
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
				rop.ll.op_unmasked_store(rop.ll.wide_constant_bool(false), loc_of_continue_mask);
            }

            rop.build_llvm_code (op.jump(1), op.jump(2), body_block);
            rop.ll.op_branch (step_block);

            // Step
            // The step shares the same mask as the body, unless a continue was called
			if (rop.ll.masked_continue_count() > 0) {
				//std::cout << "(rop.ll.masked_continue_count() > 0) == true\n";
				// Get rid of any modified mask that had the continue mask applied to it
				rop.ll.pop_mask();
				// Restore the condition mask for the step to execute with
				llvm::Value * pre_step_mask = pre_condition_mask;
				// We are trying to reuse the conditional loaded before the body
				// executes, however a 'break' would have written to that conditional mask
				// In that case, we need to reload the mask
				if (rop.ll.masked_break_count() > 0)
				{
					pre_step_mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
					// The break could have caused all lanes to be 0,
					// If there was no continue that would have jumped to the after block already.
					// But we are here because perhaps some lanes were 0 because of the continue.
					// Reloading the condition variable will not contain any continued lanes.
					// So we can test it to see if any lanes are active. If not,
					// we don't want to execute the condition block as it might contain function calls
					// or use param which calls down to subsequent layers.
					// So we will test to see if any lanes are active
					llvm::Value* anyLanesActive = rop.ll.test_if_mask_is_non_zero(pre_step_mask);
					llvm::BasicBlock* some_lanes_active_after_continue_break = rop.ll.new_basic_block (std::string("some_lanes_active_after_continue_break (varying)") + cond_name);

					rop.ll.op_branch (anyLanesActive, some_lanes_active_after_continue_break, after_block);
				}
	            rop.ll.push_mask(pre_step_mask, false /* negate */, true /* absolute */);
#ifdef __OSL_TRACE_MASKS
	            rop.llvm_print_mask("pre_step_mask");
#endif
			}
			ASSERT(op.jump(2) == op.jump(3));
			// why bother building empty step
			//rop.build_llvm_code (op.jump(2), op.jump(3), step_block);
            rop.ll.op_branch (cond_block);

            // Load the condition variable and figure out if it's nonzero
            // The step shares the same mask as the step
            rop.build_llvm_code (op.jump(0), op.jump(1), cond_block);
            rop.ll.pop_mask();
            llvm::Value* post_condition_mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
            // if a return could have been
            // executed, we need to mask out those lanes from the conditional symbol
            // because the step function would have executed with those lanes off
            // causing an endless loop
            // No need to handle break here, if encountered, it was immediately applied to the condition mask
			if (rop.ll.masked_return_count() > return_count_before_loop) {
				post_condition_mask = rop.ll.apply_return_to(post_condition_mask);
                rop.llvm_store_value (post_condition_mask, cond, /*deriv*/ 0, /*component*/ 0);
        	}


            llvm::Value* cond_val = rop.ll.test_if_mask_is_non_zero(post_condition_mask);

            // Jump to either LoopBody or AfterLoop
            rop.ll.op_branch (cond_val, body_block, after_block);

        } else {

            rop.ll.op_branch (cond_block);

            // Load the condition variable and figure out if it's nonzero
            llvm::Value* pre_condition_mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
            ASSERT(pre_condition_mask->getType() == rop.ll.type_wide_bool());
            rop.ll.push_mask(pre_condition_mask, false /* negate */, true /* absolute */);
            rop.build_llvm_code (op.jump(0), op.jump(1), cond_block);
            rop.ll.pop_mask();
            llvm::Value* post_condition_mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);

			// The condition was initializedwith the current_mask before the loop
            // and considered an absolute value, therefore should be OK to test directly
            llvm::Value* cond_val = rop.ll.test_if_mask_is_non_zero(post_condition_mask);

            // Jump to either LoopBody or AfterLoop
            rop.ll.op_branch (cond_val, body_block, after_block);

            // Body of loop
            rop.ll.push_mask(post_condition_mask, false /* negate */, true /* absolute */);
            // We need to zero out the continue mask at the top loop body, as the previous
            // iteration could have set continue, alternatively we could zero at the end
            // of the loop body so its ready for the next iteration, perhaps as part
            // of the step, but if we know we will need it simplest to do at top of loop body
            // TODO, move allocation of continue mask inside the loop body to minimize its
            // scope, although it is still a loop resource perhaps we can delay
            // setting it until now
            if (loopHasContinue) {
				rop.ll.op_unmasked_store(rop.ll.wide_constant_bool(false), loc_of_continue_mask);
            }
            rop.build_llvm_code (op.jump(1), op.jump(2), body_block);

            rop.ll.op_branch (step_block);

            // Step
            // The step shares the same mask as the body, unless a continue was called
			if (rop.ll.masked_continue_count() > 0) {
				//std::cout << "(rop.ll.masked_continue_count() > 0) == true\n";
				// Get rid of any modified mask that had the continue mask applied to it
				rop.ll.pop_mask();
				// Restore the condition mask for the step to execute with
				llvm::Value * pre_step_mask = post_condition_mask;
				// We are trying to reuse the conditional loaded before the body
				// executes, however a 'break' would have written to that conditional mask
				// In that case, we need to reload the mask
				if (rop.ll.masked_break_count() > 0)
				{
					pre_step_mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
				}
	            rop.ll.push_mask(pre_step_mask, false /* negate */, true /* absolute */);
#ifdef __OSL_TRACE_MASKS
	            rop.llvm_print_mask("pre_step_mask");
#endif
			}
            rop.build_llvm_code (op.jump(2), op.jump(3), step_block);
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
					post_condition_mask = rop.llvm_load_value (cond, /*deriv*/ 0, /*component*/ 0, /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
				}
            	llvm::Value * post_step_mask = rop.ll.apply_return_to(post_condition_mask);
                rop.llvm_store_value (post_step_mask, cond, /*deriv*/ 0, /*component*/ 0);
        	}
            rop.ll.op_branch (cond_block);

        }
        rop.ll.pop_masked_loop();
        rop.ll.pop_loop ();

        // Continue on with the previous flow
        rop.ll.set_insert_point (after_block);

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
			llvm::Value* anyLanesActive = rop.ll.test_if_mask_is_non_zero(rop.ll.current_mask());

			llvm::BasicBlock * nextMaskScope;
			if (rop.ll.has_masked_return_block()) {
				nextMaskScope = rop.ll.masked_return_block();
			} else {
				nextMaskScope = rop.ll.inside_function() ?
								rop.ll.return_block() :
								rop.llvm_exit_instance_block();
			}
	        llvm::BasicBlock* after_applying_return_block = rop.ll.new_basic_block (std::string("after_loop_applied_return_mask (varying)") + cond_name);
			rop.ll.op_branch (anyLanesActive, after_applying_return_block, nextMaskScope);
		}


    }

    return true;
}



LLVMGEN (llvm_gen_loopmod_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 0);

    bool inside_masked_loop = rop.ll.is_innermost_loop_masked();
    if (false == inside_masked_loop)
    {
        // Inside a uniform loop, can use branching
        if (op.opname() == op_break) {
            rop.ll.op_branch (rop.ll.loop_after_block());
        } else {  // continue
            rop.ll.op_branch (rop.ll.loop_step_block());
        }
        llvm::BasicBlock* next_block = rop.ll.new_basic_block ("next_block");
        rop.ll.set_insert_point (next_block);
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
        	ASSERT(op.opname() == op_continue);
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




llvm::Value* llvm_batched_texture_options(BackendLLVMWide &rop, int opnum,
                                       int first_optional_arg, bool tex3d, int nchans,
                                       llvm::Value* &alpha, llvm::Value* &dalphadx,
                                       llvm::Value* &dalphady, llvm::Value* &errormessage,
                                       llvm::Value* &missingcolor_buffer)
{
    llvm::Value * bto = rop.temp_batched_texture_options_ptr();

    // The BatchedTextureOptions & missingcolor_buffer are local alloca,
    // so no need to mask off non-active lanes

    // Explicitly assign a default value or an optional parameter value to every data member
    // of BatchedTextureOptions.
    llvm::Value * wide_const_fzero_value = rop.ll.wide_constant(0.0f);
    llvm::Value * wide_const_fone_value = rop.ll.wide_constant(1.0f);
    llvm::Value * const_zero_value = rop.ll.constant(0);
    llvm::Value * wrap_default_value = rop.ll.constant(static_cast<int>(Tex::Wrap::Default));

    llvm::Value * sblur = wide_const_fzero_value;
    llvm::Value * tblur= wide_const_fzero_value;
    llvm::Value * rblur = wide_const_fzero_value;
    llvm::Value * swidth = wide_const_fone_value;
    llvm::Value * twidth = wide_const_fone_value;
    llvm::Value * rwidth = wide_const_fone_value;

    llvm::Value * firstchannel = const_zero_value;
    llvm::Value * subimage = const_zero_value;
    llvm::Value * subimagename = rop.ll.constant_ptr(nullptr);
    llvm::Value * swrap = wrap_default_value;
    llvm::Value * twrap = wrap_default_value;
    llvm::Value * rwrap = wrap_default_value;
    llvm::Value * mipmode = rop.ll.constant(static_cast<int>(Tex::MipMode::Default));
    llvm::Value * interpmode = rop.ll.constant(static_cast<int>(Tex::InterpMode::SmartBicubic));
    llvm::Value * anisotropic = rop.ll.constant(32);
    llvm::Value * conservative_filter = rop.ll.constant(1);
    llvm::Value * fill = rop.ll.constant(0.0f);

    bool is_swrap_uniform = true;
    bool is_twrap_uniform = true;
    bool is_rwrap_uniform = true;

    bool is_fill_uniform = true;
    bool is_firstchannel_uniform = true;
    bool is_subimage_uniform = true;
    bool is_subimagename_uniform = true;
    bool is_interpmode_uniform = true;


    llvm::Value * missingcolor = rop.ll.constant_ptr(nullptr, rop.ll.type_float_ptr());

    Opcode &op (rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg;  a < op.nargs();  ++a) {
        Symbol &Name(*rop.opargsym(op,a));
        ASSERT (Name.typespec().is_string() &&
                "optional texture token must be a string");
        ASSERT (a+1 < op.nargs() && "malformed argument list for texture");
        ustring name = *(ustring *)Name.data();
        ++a;  // advance to next argument (value)

        if (! name)    // skip empty string param name
            continue;
        Symbol &Val(*rop.opargsym(op,a));
        TypeDesc valtype = Val.typespec().simpletype ();


        bool nameIsVarying = !rop.isSymbolUniform(Name);
        // assuming option names can't be varying
        ASSERT(!nameIsVarying);
        // data could be varying
        bool valIsVarying = !rop.isSymbolUniform(Val);

#define PARAM_WIDE_FLOAT(paramname)                                     \
        if (name == Strings::paramname &&                               \
            (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) { \
            llvm::Value *val = rop.llvm_load_value (Val);               \
            if (valtype == TypeDesc::INT)                               \
                val = rop.ll.op_int_to_float (val);                     \
            if (!valIsVarying) {                                        \
                val = rop.ll.widen_value(val);                          \
            }                                                           \
            paramname = val;                                            \
            continue;                                                   \
        }

#define PARAM_WIDE_FLOAT_S_T_R(paramname)                                 \
        if (name == Strings::paramname &&                                 \
            (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {   \
            llvm::Value *val = rop.llvm_load_value (Val);                 \
              if (valtype == TypeDesc::INT) {                             \
                  val = rop.ll.op_int_to_float (val);                     \
              }                                                           \
              if (!valIsVarying) {                                        \
                  val = rop.ll.widen_value(val);                          \
              }                                                           \
              s##paramname = val;                                         \
              t##paramname = val;                                         \
              if (tex3d) {                                                \
                  r##paramname = val;                                     \
              }                                                           \
            continue;                                                     \
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


#define PARAM_UNIFORM_FLOAT(paramname)                                  \
        if (name == Strings::paramname &&                               \
            (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) { \
            is_##paramname##_uniform = !valIsVarying;                   \
            if (valIsVarying) {                                         \
                continue;                                               \
            }                                                           \
            llvm::Value *val = rop.llvm_load_value (Val);               \
            if (valtype == TypeDesc::INT)                               \
                val = rop.ll.op_int_to_float (val);                     \
            paramname = val;                                            \
            continue;                                                   \
        }

#define PARAM_UNIFORM_INT(paramname)                                    \
        if (name == Strings::paramname &&                               \
            (valtype == TypeDesc::INT)) {                               \
            is_##paramname##_uniform = !valIsVarying;                   \
            if (valIsVarying) {                                         \
                continue;                                               \
            }                                                           \
            llvm::Value *val = rop.llvm_load_value (Val);               \
            paramname = val;                                            \
            continue;                                                   \
        }

#define __OSL_STRINGIFY(val) #val

#define PARAM_UNIFORM_STRING_CODE(paramname,decoder, llvm_decoder, fieldname)                 \
        if (name == Strings::paramname && valtype == TypeDesc::STRING) {           \
            if (valIsVarying) {                                                    \
                is_##fieldname##_uniform = false;                                  \
                continue;                                                          \
            }                                                                      \
            llvm::Value *val = nullptr;                                            \
            if (Val.is_constant()) {                                               \
                int mode = decoder (*(ustring *)Val.data());                       \
                val = rop.ll.constant (mode);                                      \
            } else {                                                               \
                val = rop.llvm_load_value (Val);                                   \
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
            llvm::Value *val = nullptr;
            if (Val.is_constant()) {
                int mode = TextureOpt::decode_wrapmode (*(ustring *)Val.data());
                val = rop.ll.constant (mode);
            } else {
                val = rop.llvm_load_value (Val);
                val = rop.ll.call_function("osl_texture_decode_wrapmode", val);
            }
            swrap = val;
            twrap = val;
            if (tex3d) {
                rwrap = val;
            }
            continue;
        }
        PARAM_UNIFORM_STRING_CODE(swrap, OIIO::TextureOpt::decode_wrapmode, osl_texture_decode_wrapmode, swrap)
        PARAM_UNIFORM_STRING_CODE(twrap, OIIO::TextureOpt::decode_wrapmode, osl_texture_decode_wrapmode, twrap)
        if (tex3d) {
            PARAM_UNIFORM_STRING_CODE(rwrap, OIIO::TextureOpt::decode_wrapmode, osl_texture_decode_wrapmode, rwrap)
        }

        PARAM_UNIFORM_FLOAT(fill)
        PARAM_UNIFORM_INT(firstchannel)
        PARAM_UNIFORM_INT(subimage)

        if (name == Strings::subimage && valtype == TypeDesc::STRING) {
            if (valIsVarying) {
                is_subimagename_uniform = false;
                continue;
            }
            llvm::Value *val = rop.llvm_load_value (Val);
            subimagename = val;
            continue;
        }

        PARAM_UNIFORM_STRING_CODE(interp, tex_interp_to_code, osl_texture_decode_interpmode, interpmode)


        if (name == Strings::alpha && valtype == TypeDesc::FLOAT) {
            ASSERT(valIsVarying && "thought to be unreachable as texture is inherently wide, so any alpha variable needs to be as well");
            // We will handle this as part of the uniform section, as we will point to wide variables
            // and technically the alpha is part of the output parameters not a texture option
            alpha = rop.llvm_get_pointer (Val);
            if (Val.has_derivs()) {
                dalphadx = rop.llvm_get_pointer (Val, 1);
                dalphady = rop.llvm_get_pointer (Val, 2);
                // NO z derivs!  dalphadz = rop.llvm_get_pointer (Val, 3);
            }
            continue;
        }

        if (name == Strings::errormessage && valtype == TypeDesc::STRING) {
            ASSERT(valIsVarying && "thought to be unreachable as texture is inherently wide, so any errormessage variable needs to be as well");
            // We will handle this as part of the uniform section, as we will point to wide variables
            // and technically the alpha is part of the output parameters not a texture option
            errormessage = rop.llvm_get_pointer (Val);
            continue;
        }

        if (name == Strings::missingcolor &&
                   equivalent(valtype,TypeDesc::TypeColor)) {
            if (! missingcolor_buffer) {
                // If not already done, allocate enough storage for the
                // missingcolor value (4 floats), and call the special
                // function that points the TextureOpt.missingcolor to it.
                missingcolor_buffer = rop.ll.op_alloca(rop.ll.type_float(), 4, "float missingcolor[4]");
                missingcolor = missingcolor_buffer;
            }
            if (valIsVarying) {
                // For the varying case, we still wanted to allocate a scalar missingcolor_buffer
                // and point the uniform portion of the BatchedTextureOptions at it
                // So we don't track if its uniform, as we always write it out
                continue;
            }
            ASSERT(missingcolor_buffer != nullptr);
            rop.ll.op_memcpy (rop.ll.void_ptr(missingcolor_buffer),
                              rop.llvm_void_ptr(Val), (int)sizeof(Color3));
            continue;
        }

        if (name == Strings::missingalpha &&
                   valtype == TypeDesc::FLOAT) {
            if (! missingcolor_buffer) {
                // If not already done, allocate enough storage for the
                // missingcolor value (4 floats), and call the special
                // function that points the TextureOpt.missingcolor to it.
                missingcolor_buffer = rop.ll.op_alloca(rop.ll.type_float(), 4, "float missingcolor[4]");
                missingcolor = missingcolor_buffer;
            }
            if (valIsVarying) {
                // For the varying case, we still wanted to allocate a scalar missingcolor_buffer
                // and point the uniform portion of the BatchedTextureOptions at it
                // So we don't track if its uniform, as we always write it out
                continue;
            }
            ASSERT(missingcolor_buffer != nullptr);
            llvm::Value *val = rop.llvm_load_value (Val);
            // Depending on how render services handles channels, might need to be 3 period.
            rop.ll.op_unmasked_store (val, rop.ll.GEP(missingcolor_buffer, 3/*nchans*/));
            continue;
        }

        rop.shadingcontext()->error ("Unknown texture%s optional argument: \"%s\", <%s> (%s:%d)",
                                     tex3d ? "3d" : "",
                                     name.c_str(), valtype.c_str(),
                                     op.sourcefile().c_str(), op.sourceline());

#undef PARAM_WIDE_FLOAT
#undef PARAM_WIDE_FLOAT_S_T_R
#undef PARAM_UNIFORM_FLOAT
#undef PARAM_UNIFORM_INT
#undef PARAM_UNIFORM_STRING_CODE
    }

    if (is_firstchannel_uniform)
        rop.ll.op_unmasked_store (firstchannel, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::firstchannel)));
    if (is_subimage_uniform)
        rop.ll.op_unmasked_store (subimage, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::subimage)));

    if (is_subimagename_uniform)
        rop.ll.op_unmasked_store (subimagename, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::subimagename)));

    if (is_swrap_uniform)
        rop.ll.op_unmasked_store (swrap, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::swrap)));
    if (is_twrap_uniform)
        rop.ll.op_unmasked_store (twrap, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::twrap)));
    if (is_rwrap_uniform)
        rop.ll.op_unmasked_store (rwrap, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::rwrap)));

    // No way to set mipmode option from OSL
    rop.ll.op_unmasked_store (mipmode, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::mipmode)));

    if (is_interpmode_uniform)
        rop.ll.op_unmasked_store (interpmode, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::interpmode)));

    // No way to set anisotropic option from OSL
    rop.ll.op_unmasked_store (anisotropic, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::anisotropic)));

    // No way to set conservative_filter option from OSL
    rop.ll.op_unmasked_store (conservative_filter, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::conservative_filter)));

    if (is_fill_uniform)
        rop.ll.op_unmasked_store (fill, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::fill)));

    // For uniform and varying we point the missingcolor to nullptr or the missingcolor_buffer,
    // The varying options will copy the lead lane's missing color value into the missingcolor_buffer
    rop.ll.op_unmasked_store (missingcolor, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::missingcolor)));

    // blur's and width's are always communicated as wide, we we will handle them here
    rop.ll.op_unmasked_store (sblur, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::sblur)));
    rop.ll.op_unmasked_store (tblur, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::tblur)));
    rop.ll.op_unmasked_store (swidth, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::swidth)));
    rop.ll.op_unmasked_store (twidth, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::twidth)));

    if (tex3d) {
            rop.ll.op_unmasked_store (rblur, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::rblur)));
            rop.ll.op_unmasked_store (rwidth, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::rwidth)));
    }

    return rop.ll.void_ptr(bto);

}


llvm::Value* llvm_batched_texture_varying_options(BackendLLVMWide &rop, int opnum,
                                       int first_optional_arg, bool tex3d, int nchans, llvm::Value *remainingMask,
                                       llvm::Value * leadLane,
                                       llvm::Value* missingcolor_buffer)
{
    llvm::Value * bto = rop.temp_batched_texture_options_ptr();

    // The BatchedTextureOptions & missingcolor_buffer are local alloca,
    // so no need to mask off non-active lanes

    Opcode &op (rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg;  a < op.nargs();  ++a) {
        Symbol &Name(*rop.opargsym(op,a));
        ASSERT (Name.typespec().is_string() &&
                "optional texture token must be a string");
        ASSERT (a+1 < op.nargs() && "malformed argument list for texture");
        ustring name = *(ustring *)Name.data();
        ++a;  // advance to next argument (value)

        if (! name)    // skip empty string param name
            continue;
        Symbol &Val(*rop.opargsym(op,a));
        TypeDesc valtype = Val.typespec().simpletype ();


        bool nameIsVarying = !rop.isSymbolUniform(Name);
        // assuming option names can't be varying
        ASSERT(!nameIsVarying);

        // data could be uniform
        bool valIsVarying = !rop.isSymbolUniform(Val);
        if (!valIsVarying)
            continue;

        ASSERT(!Val.is_constant() && "can't be a varying constant");

#define SKIP_PARAM_WIDE_FLOAT(paramname)                                \
        if (name == Strings::paramname &&                               \
            (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) { \
            continue;                                                   \
        }

#define SKIP_PARAM_WIDE_STRING(paramname)                                \
        if (name == Strings::paramname &&                               \
            valtype == TypeDesc::STRING) { \
            continue;                                                   \
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
            llvm::Value *wide_wrap = rop.llvm_load_value (Val);
            llvm::Value *scalar_value = rop.ll.op_extract(wide_wrap, leadLane);
            llvm::Value *wrap_code = rop.ll.call_function("osl_texture_decode_wrapmode", scalar_value);
            rop.ll.op_unmasked_store (wrap_code, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::swrap)));
            rop.ll.op_unmasked_store (wrap_code, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::twrap)));

            if (tex3d)
                rop.ll.op_unmasked_store (wrap_code, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::rwrap)));

            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value, wide_wrap, remainingMask);
            continue;
        }

#define PARAM_VARYING_STRING_CODE(paramname, llvm_decoder, fieldname)                 \
        if (name == Strings::paramname && valtype == TypeDesc::STRING) {                      \
            llvm::Value *wide_value = rop.llvm_load_value (Val);                              \
            llvm::Value *scalar_value = rop.ll.op_extract(wide_value, leadLane);              \
            llvm::Value *scalar_code = rop.ll.call_function(#llvm_decoder, scalar_value); \
            rop.ll.op_unmasked_store (scalar_code, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::fieldname))); \
            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value, wide_value, remainingMask); \
            continue;                                                                         \
        }

        PARAM_VARYING_STRING_CODE(swrap, osl_texture_decode_wrapmode, swrap)
        PARAM_VARYING_STRING_CODE(twrap, osl_texture_decode_wrapmode, twrap)
        if (tex3d) {
            PARAM_VARYING_STRING_CODE(rwrap, osl_texture_decode_wrapmode, rwrap)
        }


        if (name == Strings::fill && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            llvm::Value *wide_val = rop.llvm_load_value (Val);
            llvm::Value *scalar_value = rop.ll.op_extract(wide_val, leadLane);
            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value, wide_val, remainingMask);
            if (valtype == TypeDesc::INT)
                scalar_value = rop.ll.op_int_to_float (scalar_value);
            rop.ll.op_unmasked_store (scalar_value, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::fill)));
            continue;
        }

#define PARAM_VARYING(paramname, paramtype, fieldname)                 \
        if (name == Strings::paramname && valtype == paramtype) {      \
            llvm::Value *wide_val = rop.llvm_load_value (Val);         \
            llvm::Value *scalar_value = rop.ll.op_extract(wide_val, leadLane); \
            rop.ll.op_unmasked_store (scalar_value, rop.ll.GEP (bto, 0, static_cast<int>(BatchedTextureOptions::LLVMMemberIndex::fieldname))); \
            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value, wide_val, remainingMask); \
            continue; \
        }
        PARAM_VARYING(firstchannel, TypeDesc::INT, firstchannel)
        PARAM_VARYING(subimage, TypeDesc::INT, subimage)
        PARAM_VARYING(subimage, TypeDesc::STRING, subimagename)

        PARAM_VARYING_STRING_CODE(interp, osl_texture_decode_interpmode, interpmode)

        if (name == Strings::missingcolor &&
                   equivalent(valtype,TypeDesc::TypeColor)) {
            ASSERT(missingcolor_buffer != nullptr);

            int num_components = valtype.aggregate;
            for (int i = 0; i < num_components; i++) {
                llvm::Value *wide_component = rop.llvm_load_value (Val, 0, i, valtype, false /*op_is_uniform*/);
                llvm::Value *scalar_component = rop.ll.op_extract(wide_component, leadLane);
                // The missingcolor_buffer is a local alloca, so no need to mask off non-active lanes
                rop.ll.op_unmasked_store (scalar_component, rop.ll.GEP(missingcolor_buffer,i));

                remainingMask = rop.ll.op_lanes_that_match_masked(scalar_component, wide_component, remainingMask);
            }
            continue;
        }

        if (name == Strings::missingalpha && valtype == TypeDesc::FLOAT) {
            ASSERT(missingcolor_buffer != nullptr);

            llvm::Value *wide_missingalpha = rop.llvm_load_value (Val);
            llvm::Value *scalar_missingalpha = rop.ll.op_extract(wide_missingalpha, leadLane);
            // Depending on how render services handles channels, might need to be 3 period.
            rop.ll.op_unmasked_store (scalar_missingalpha, rop.ll.GEP(missingcolor_buffer, 3 /*nchans*/));

            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_missingalpha, wide_missingalpha, remainingMask);
            continue;
        }

        rop.shadingcontext()->error ("Unknown texture%s optional argument: \"%s\", <%s> (%s:%d)",
                                     tex3d ? "3d" : "",
                                     name.c_str(), valtype.c_str(),
                                     op.sourcefile().c_str(), op.sourceline());

#undef SKIP_PARAM_WIDE_FLOAT
#undef SKIP_PARAM_WIDE_STRING
#undef PARAM_VARYING_STRING_CODE
#undef PARAM_VARYING

    }

    return remainingMask;
}

LLVMGEN (llvm_gen_texture)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Filename = *rop.opargsym (op, 1);
    Symbol &S = *rop.opargsym (op, 2);
    Symbol &T = *rop.opargsym (op, 3);
    int nchans = Result.typespec().aggregate();

    bool user_derivs = false;
    int first_optional_arg = 4;
    if (op.nargs() > 4 && rop.opargsym(op,4)->typespec().is_float()) {
        user_derivs = true;
        first_optional_arg = 8;
        DASSERT (rop.opargsym(op,5)->typespec().is_float());
        DASSERT (rop.opargsym(op,6)->typespec().is_float());
        DASSERT (rop.opargsym(op,7)->typespec().is_float());
    }

    llvm::Value* opt;   // TextureOpt
    llvm::Value *alpha = NULL, *dalphadx = NULL, *dalphady = NULL;
    llvm::Value *errormessage = NULL;

    llvm::Value* missingcolor_buffer = nullptr;
    opt = llvm_batched_texture_options (rop, opnum, first_optional_arg,
                                    false /*3d*/, nchans,
                                    alpha, dalphadx, dalphady, errormessage,
                                    missingcolor_buffer);

    // Now call the osl_texture function, passing the options and all the
    // explicit args like texture coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());

    bool fileNameIsUniform = rop.isSymbolUniform(Filename);
    ustring texFuncName("osl_texture_batched");
    RendererServices::TextureHandle *texture_handle = NULL;
    if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        ASSERT(fileNameIsUniform);
        texture_handle = rop.renderer()->get_texture_handle (*(ustring *)Filename.data());
        if (! rop.renderer()->good (texture_handle))
            texture_handle = NULL;
    }

    // We will just load the filename here if we are uniform, otherwise just remember where the filename
    // is so we can update it later in the loop over varying options
    int filenameArgumentIndex = args.size();
    llvm::Value * filenameVal = rop.llvm_load_value (Filename);
    args.push_back (fileNameIsUniform ? filenameVal : nullptr);

    args.push_back (rop.ll.constant_ptr (texture_handle));
    rop.generated_texture_call (texture_handle != NULL);

    // check S & T are not uniform

    llvm::Value* wideS = nullptr;
    llvm::Value* wideSD1 = nullptr;
    llvm::Value* wideSD2 = nullptr;
    llvm::Value* wideT = nullptr;
    llvm::Value* wideTD1 = nullptr;
    llvm::Value* wideTD2 = nullptr;

    if (rop.isSymbolUniform(S)) {
        wideS = rop.llvm_alloca_and_widen_value(S, 0);
        if (!user_derivs) {
            wideSD1 = rop.llvm_alloca_and_widen_value(S, 1);
            wideSD2 = rop.llvm_alloca_and_widen_value(S, 2);
        }
    }
    else {
        wideS = rop.llvm_void_ptr(S, 0);
        if (!user_derivs) {
            wideSD1 = rop.llvm_void_ptr(S, 1);
            wideSD2 = rop.llvm_void_ptr(S, 2);
        }
    }
    if (rop.isSymbolUniform(T)) {
        wideT = rop.llvm_alloca_and_widen_value(S, 0);
        if (!user_derivs) {
            wideTD1 = rop.llvm_alloca_and_widen_value(S, 1);
            wideTD2 = rop.llvm_alloca_and_widen_value(S, 2);
        }
    }
    else {
        wideT = rop.llvm_void_ptr(T);
        if (!user_derivs) {
            wideTD1 = rop.llvm_void_ptr(T, 1);
            wideTD2 = rop.llvm_void_ptr(T, 2);
        }
    }
    args.push_back (opt);
    args.push_back (wideS);
    args.push_back (wideT);
    llvm::Value* wideDsDx = nullptr;
    llvm::Value* wideDtDx = nullptr;
    llvm::Value* wideDsDy = nullptr;
    llvm::Value* wideDtDy = nullptr;

    if (user_derivs) {
        Symbol &DsDx = *rop.opargsym (op, 4);
        Symbol &DtDx = *rop.opargsym (op, 5);
        Symbol &DsDy = *rop.opargsym (op, 6);
        Symbol &DtDy = *rop.opargsym (op, 7);
        if (rop.isSymbolUniform(DsDx)) {
            wideDsDx = rop.llvm_alloca_and_widen_value(DsDx, 0);
        } else {
            wideDsDx = rop.llvm_void_ptr(DsDx, 0);
        }
        if (rop.isSymbolUniform(DtDx)) {
            wideDtDx = rop.llvm_alloca_and_widen_value(DtDx, 0);
        } else {
            wideDtDx = rop.llvm_void_ptr(DtDx, 0);
        }
        if (rop.isSymbolUniform(DsDy)) {
            wideDsDy = rop.llvm_alloca_and_widen_value(DsDy, 0);
        } else {
            wideDsDy = rop.llvm_void_ptr(DsDy, 0);
        }
        if (rop.isSymbolUniform(DtDy)) {
            wideDtDy = rop.llvm_alloca_and_widen_value(DtDy, 0);
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
    args.push_back (wideDsDx);
    args.push_back (wideDtDx);
    args.push_back (wideDsDy);
    args.push_back (wideDtDy);

    OSL_DEV_ONLY(std::cout << "texture result type: " << rop.ll.llvm_typenameof(rop.llvm_get_pointer (Result, 1)) << std::endl);
    args.push_back (rop.ll.constant (nchans));
    args.push_back (rop.ll.void_ptr (rop.llvm_get_pointer (Result, 0)));
    args.push_back (Result.has_derivs() ? rop.ll.constant(1) : rop.ll.constant(0));
    args.push_back (rop.ll.void_ptr (alpha    ? alpha    : rop.ll.void_ptr_null()));
    args.push_back ((dalphadx && dalphady) ? rop.ll.constant(1) : rop.ll.constant(0));
    args.push_back (rop.ll.void_ptr (errormessage ? errormessage : rop.ll.void_ptr_null()));

    // do while(remaining)
    llvm::Value * loc_of_remainingMask = rop.ll.op_alloca (rop.ll.type_wide_bool(), 1, "lanes remaining to texture");
    rop.ll.op_unmasked_store(rop.ll.current_mask(), loc_of_remainingMask);

    llvm::BasicBlock* bin_block = rop.ll.new_basic_block (std::string("bin_texture_options (varying texture options)"));
    llvm::BasicBlock* after_block = rop.ll.new_basic_block (std::string("after_bin_texture_options (varying texture options)"));
    rop.ll.op_branch(bin_block);
    {

        llvm::Value * remainingMask = rop.ll.op_load(loc_of_remainingMask);
        llvm::Value * leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
        llvm::Value * lanesMatchingFilename = remainingMask;

        if(false == fileNameIsUniform) {
            llvm::Value *scalar_filename = rop.ll.op_extract(filenameVal, leadLane);
            args[filenameArgumentIndex] = scalar_filename;
            lanesMatchingFilename = rop.ll.op_lanes_that_match_masked(scalar_filename, filenameVal, remainingMask);
        }

        //rop.llvm_print_mask("before remainingMask", remainingMask);
        llvm::Value * lanesMatchingOptions = llvm_batched_texture_varying_options (rop, opnum, first_optional_arg,
                                        false /*3d*/, nchans, lanesMatchingFilename, leadLane, missingcolor_buffer);
        ASSERT(lanesMatchingOptions);
        //rop.llvm_print_mask("lanesMatchingOptions", lanesMatchingOptions);
        args.push_back (rop.ll.mask_as_int(lanesMatchingOptions));

        rop.ll.call_function (texFuncName.c_str(), &args[0], (int)args.size());

        remainingMask = rop.ll.op_xor(remainingMask,lanesMatchingOptions);
        //rop.llvm_print_mask("xor remainingMask,lanesMatchingOptions", remainingMask);
        rop.ll.op_unmasked_store(remainingMask, loc_of_remainingMask);

        llvm::Value * int_remainingMask = rop.ll.mask_as_int(remainingMask);
        //rop.llvm_print_mask("remainingMask", remainingMask);
        llvm::Value* cond_more_lanes_to_bin = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
        rop.ll.op_branch (cond_more_lanes_to_bin, bin_block, after_block);
    }
    // Continue on with the previous flow
    rop.ll.set_insert_point (after_block);

    return true;
}


LLVMGEN (llvm_gen_texture3d)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Filename = *rop.opargsym (op, 1);
    Symbol &P = *rop.opargsym (op, 2);
    int nchans = Result.typespec().aggregate();

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    bool user_derivs = false;
    int first_optional_arg = 3;
    if (op.nargs() > 3 && rop.opargsym(op,3)->typespec().is_triple()) {
        user_derivs = true;
        first_optional_arg = 5;
        DASSERT (rop.opargsym(op,3)->typespec().is_triple());
        DASSERT (rop.opargsym(op,4)->typespec().is_triple());
    }

    llvm::Value* opt;   // TextureOpt
    llvm::Value *alpha = NULL, *dalphadx = NULL, *dalphady = NULL;
    llvm::Value *errormessage = NULL;
//    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
//                                    true /*3d*/, nchans,
//                                    alpha, dalphadx, dalphady, errormessage);

    // Now call the osl_texture3d function, passing the options and all the
    // explicit args like texture coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    RendererServices::TextureHandle *texture_handle = NULL;
    if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        texture_handle = rop.renderer()->get_texture_handle (*(ustring *)Filename.data());
        if (! rop.renderer()->good (texture_handle))
            texture_handle = NULL;
    }
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (rop.ll.constant_ptr (texture_handle));
    args.push_back (opt);
    args.push_back (rop.llvm_void_ptr (P));
    if (user_derivs) {
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 3)));
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 4)));
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 5)));
    } else {
        // Auto derivs of P
        args.push_back (rop.llvm_void_ptr (P, 1));
        args.push_back (rop.llvm_void_ptr (P, 2));
        // dPdz is correct for input P, zero for all else
        if (&P == rop.inst()->symbol(rop.inst()->Psym())) {
            args.push_back (rop.llvm_void_ptr (P, 3));
        } else {
            // zero for dPdz, for now
            llvm::Value *fzero = rop.ll.constant (0.0f);
            llvm::Value *vzero = rop.ll.op_alloca (rop.ll.type_triple(), 1, "vzero");
            for (int i = 0;  i < 3;  ++i)
                rop.ll.op_store (fzero, rop.ll.GEP (vzero, 0, i));
            args.push_back (rop.ll.void_ptr(vzero));
        }
    }
    args.push_back (rop.ll.constant (nchans));
    args.push_back (rop.ll.void_ptr (rop.llvm_void_ptr (Result, 0)));
    args.push_back (rop.ll.void_ptr (rop.llvm_void_ptr (Result, 1)));
    args.push_back (rop.ll.void_ptr (rop.llvm_void_ptr (Result, 2)));
    args.push_back (rop.ll.void_ptr_null());  // no dresultdz for now
    args.push_back (rop.ll.void_ptr (alpha    ? alpha    : rop.ll.void_ptr_null()));
    args.push_back (rop.ll.void_ptr (dalphadx ? dalphadx : rop.ll.void_ptr_null()));
    args.push_back (rop.ll.void_ptr (dalphady ? dalphady : rop.ll.void_ptr_null()));
    args.push_back (rop.ll.void_ptr_null());  // No dalphadz for now
    args.push_back (rop.ll.void_ptr (errormessage ? errormessage : rop.ll.void_ptr_null()));
    rop.ll.call_function ("osl_texture3d", &args[0], (int)args.size());
    rop.generated_texture_call (texture_handle != NULL);
    return true;
}



LLVMGEN (llvm_gen_environment)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Filename = *rop.opargsym (op, 1);
    Symbol &R = *rop.opargsym (op, 2);
    int nchans = Result.typespec().aggregate();

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    bool user_derivs = false;
    int first_optional_arg = 3;
    if (op.nargs() > 3 && rop.opargsym(op,3)->typespec().is_triple()) {
        user_derivs = true;
        first_optional_arg = 5;
        DASSERT (rop.opargsym(op,4)->typespec().is_triple());
    }

    llvm::Value* opt;   // TextureOpt
    llvm::Value *alpha = NULL, *dalphadx = NULL, *dalphady = NULL;
    llvm::Value *errormessage = NULL;
//    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
//                                    false /*3d*/, nchans,
//                                    alpha, dalphadx, dalphady, errormessage);

    // Now call the osl_environment function, passing the options and all the
    // explicit args like texture coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    RendererServices::TextureHandle *texture_handle = NULL;
    if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        texture_handle = rop.renderer()->get_texture_handle (*(ustring *)Filename.data());
        if (! rop.renderer()->good (texture_handle))
            texture_handle = NULL;
    }
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (rop.ll.constant_ptr (texture_handle));
    args.push_back (opt);
    args.push_back (rop.llvm_void_ptr (R));
    if (user_derivs) {
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 3)));
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 4)));
    } else {
        // Auto derivs of R
        args.push_back (rop.llvm_void_ptr (R, 1));
        args.push_back (rop.llvm_void_ptr (R, 2));
    }
    args.push_back (rop.ll.constant (nchans));
    args.push_back (rop.llvm_void_ptr (Result, 0));
    args.push_back (rop.llvm_void_ptr (Result, 1));
    args.push_back (rop.llvm_void_ptr (Result, 2));
    if (alpha) {
        args.push_back (rop.ll.void_ptr (alpha));
        args.push_back (dalphadx ? rop.ll.void_ptr (dalphadx) : rop.ll.void_ptr_null());
        args.push_back (dalphady ? rop.ll.void_ptr (dalphady) : rop.ll.void_ptr_null());
    } else {
        args.push_back (rop.ll.void_ptr_null());
        args.push_back (rop.ll.void_ptr_null());
        args.push_back (rop.ll.void_ptr_null());
    }
    args.push_back (rop.ll.void_ptr (errormessage ? errormessage : rop.ll.void_ptr_null()));
    rop.ll.call_function ("osl_environment", &args[0], (int)args.size());
    rop.generated_texture_call (texture_handle != NULL);
    return true;
}



static llvm::Value *
llvm_gen_trace_options (BackendLLVMWide &rop, int opnum,
                        int first_optional_arg)
{
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched
    llvm::Value* opt = rop.ll.call_function ("osl_get_trace_options",
                                             rop.sg_void_ptr());
    Opcode &op (rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg;  a < op.nargs();  ++a) {
        Symbol &Name (*rop.opargsym(op,a));
        ASSERT (Name.typespec().is_string() &&
                "optional trace token must be a string");
        ASSERT (a+1 < op.nargs() && "malformed argument list for trace");
        ustring name = *(ustring *)Name.data();

        ++a;  // advance to next argument
        Symbol &Val (*rop.opargsym(op,a));
        TypeDesc valtype = Val.typespec().simpletype ();

        llvm::Value *val = rop.llvm_load_value (Val);
        static ustring kmindist("mindist"), kmaxdist("maxdist");
        static ustring kshade("shade"), ktraceset("traceset");
        if (name == kmindist && valtype == TypeDesc::FLOAT) {
            rop.ll.call_function ("osl_trace_set_mindist", opt, val);
        } else if (name == kmaxdist && valtype == TypeDesc::FLOAT) {
            rop.ll.call_function ("osl_trace_set_maxdist", opt, val);
        } else if (name == kshade && valtype == TypeDesc::INT) {
            rop.ll.call_function ("osl_trace_set_shade", opt, val);
        } else if (name == ktraceset && valtype == TypeDesc::STRING) {
            rop.ll.call_function ("osl_trace_set_traceset", opt, val);
        } else {
            rop.shadingcontext()->error ("Unknown trace() optional argument: \"%s\", <%s> (%s:%d)",
                                    name.c_str(), valtype.c_str(),
                                    op.sourcefile().c_str(), op.sourceline());
        }
    }

    return opt;
}



LLVMGEN (llvm_gen_trace)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Pos = *rop.opargsym (op, 1);
    Symbol &Dir = *rop.opargsym (op, 2);
    int first_optional_arg = 3;

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value* opt;   // TraceOpt
    opt = llvm_gen_trace_options (rop, opnum, first_optional_arg);

    // Now call the osl_trace function, passing the options and all the
    // explicit args like trace coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (opt);
    args.push_back (rop.llvm_void_ptr (Pos, 0));
    args.push_back (rop.llvm_void_ptr (Pos, 1));
    args.push_back (rop.llvm_void_ptr (Pos, 2));
    args.push_back (rop.llvm_void_ptr (Dir, 0));
    args.push_back (rop.llvm_void_ptr (Dir, 1));
    args.push_back (rop.llvm_void_ptr (Dir, 2));
    llvm::Value *r = rop.ll.call_function ("osl_trace", &args[0],
                                             (int)args.size());
    rop.llvm_store_value (r, Result);
    return true;
}





static llvm::Value *
llvm_gen_noise_options (BackendLLVMWide &rop, int opnum,
                        int first_optional_arg)
{
    llvm::Value* opt = rop.ll.call_function ("osl_wide_get_noise_options",
                                             rop.sg_void_ptr());
    Opcode &op (rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg;  a < op.nargs();  ++a) {
        Symbol &Name (*rop.opargsym(op,a));
        ASSERT (Name.typespec().is_string() &&
                "optional noise token must be a string");
        ASSERT (a+1 < op.nargs() && "malformed argument list for noise");
        ustring name = *(ustring *)Name.data();

        ++a;  // advance to next argument
        Symbol &Val (*rop.opargsym(op,a));
        TypeDesc valtype = Val.typespec().simpletype ();

        if (! name)    // skip empty string param name
            continue;

        if (name == Strings::anisotropic && Val.typespec().is_int()) {
            rop.ll.call_function ("osl_noiseparams_set_anisotropic", opt,
                                    rop.llvm_load_value (Val));
        } else if (name == Strings::do_filter && Val.typespec().is_int()) {
            rop.ll.call_function ("osl_noiseparams_set_do_filter", opt,
                                    rop.llvm_load_value (Val));
        } else if (name == Strings::direction && Val.typespec().is_triple()) {
            rop.ll.call_function ("osl_noiseparams_set_direction", opt,
                                    rop.llvm_void_ptr (Val));
        } else if (name == Strings::bandwidth &&
                   (Val.typespec().is_float() || Val.typespec().is_int())) {
            rop.ll.call_function ("osl_noiseparams_set_bandwidth", opt,
                                    rop.llvm_load_value (Val, 0, NULL, 0,
                                                         TypeDesc::TypeFloat));
        } else if (name == Strings::impulses &&
                   (Val.typespec().is_float() || Val.typespec().is_int())) {
            rop.ll.call_function ("osl_noiseparams_set_impulses", opt,
                                    rop.llvm_load_value (Val, 0, NULL, 0,
                                                         TypeDesc::TypeFloat));
        } else {
            rop.shadingcontext()->error ("Unknown %s optional argument: \"%s\", <%s> (%s:%d)",
                                    op.opname().c_str(),
                                    name.c_str(), valtype.c_str(),
                                    op.sourcefile().c_str(), op.sourceline());
        }
    }
    return opt;
}



// T noise ([string name,] float s, ...);
// T noise ([string name,] float s, float t, ...);
// T noise ([string name,] point P, ...);
// T noise ([string name,] point P, float t, ...);
// T pnoise ([string name,] float s, float sper, ...);
// T pnoise ([string name,] float s, float t, float sper, float tper, ...);
// T pnoise ([string name,] point P, point Pper, ...);
// T pnoise ([string name,] point P, float t, point Pper, float tper, ...);
LLVMGEN (llvm_gen_noise)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    bool periodic = (op.opname() == Strings::pnoise ||
                     op.opname() == Strings::psnoise);

    int arg = 0;   // Next arg to read
    Symbol &Result = *rop.opargsym (op, arg++);

    bool op_is_uniform = rop.isSymbolUniform(Result);
    OSL_DEV_ONLY(std::cout << "llvm_gen_noise op_is_uniform="<<op_is_uniform<< std::endl);

    //int outdim =  Result.typespec().is_triple() ? 3 : 1;
    Symbol *Name = rop.opargsym (op, arg++);
    ustring name;
    if (Name->typespec().is_string()) {
        name = Name->is_constant() ? *(ustring *)Name->data() : ustring();
    } else {
        // Not a string, must be the old-style noise/pnoise
        --arg;  // forget that arg
        Name = NULL;
        name = op.opname();
    }

    Symbol *S = rop.opargsym (op, arg++), *T = NULL;
    Symbol *Sper = NULL, *Tper = NULL;
    int indim = S->typespec().is_triple() ? 3 : 1;
    bool derivs = S->has_derivs();

    if (periodic) {
        if (op.nargs() > (arg+1) &&
                (rop.opargsym(op,arg+1)->typespec().is_float() ||
                 rop.opargsym(op,arg+1)->typespec().is_triple())) {
            // 2D or 4D
            ++indim;
            T = rop.opargsym (op, arg++);
            derivs |= T->has_derivs();
        }
        Sper = rop.opargsym (op, arg++);
        if (indim == 2 || indim == 4)
            Tper = rop.opargsym (op, arg++);
    } else {
        // non-periodic case
        if (op.nargs() > arg && rop.opargsym(op,arg)->typespec().is_float()) {
            // either 2D or 4D, so needs a second index
            ++indim;
            T = rop.opargsym (op, arg++);
            derivs |= T->has_derivs();
        }
    }
    derivs &= Result.has_derivs();  // ignore derivs if result doesn't need

    bool pass_name = false, pass_sg = false, pass_options = false;
    //bool pass_mask = false;
    if (! name) {
        // name is not a constant
        name = periodic ? Strings::genericpnoise : Strings::genericnoise;
        pass_name = true;
        pass_sg = true;
        pass_options = true;
        derivs = true;   // always take derivs if we don't know noise type
    } else if (name == Strings::perlin || name == Strings::snoise ||
               name == Strings::psnoise) {
        name = periodic ? Strings::psnoise : Strings::snoise;
        // derivs = false;
    } else if (name == Strings::uperlin || name == Strings::noise ||
               name == Strings::pnoise) {
        name = periodic ? Strings::pnoise : Strings::noise;
        // derivs = false;
    } else if (name == Strings::cell || name == Strings::cellnoise) {
        name = periodic ? Strings::pcellnoise : Strings::cellnoise;
        derivs = false;  // cell noise derivs are always zero
    } else if (name == Strings::simplex && !periodic) {
        name = Strings::simplexnoise;
    } else if (name == Strings::usimplex && !periodic) {
        name = Strings::usimplexnoise;
    } else if (name == Strings::gabor) {
        // already named
        pass_name = true;
        pass_sg = true;
        pass_options = true;
        // Incomplete, adding masking support for expensive noise calls
        // or calls that might want to change direction of vectorization
        // making use of the mask to avoid work.
        //pass_mask = true;
        derivs = true;
        name = periodic ? Strings::gaborpnoise : Strings::gabornoise;
    } else {
        rop.shadingcontext()->error ("%snoise type \"%s\" is unknown, called from (%s:%d)",
                                (periodic ? "periodic " : ""), name.c_str(),
                                op.sourcefile().c_str(), op.sourceline());
        return false;
    }

    if (rop.shadingsys().no_noise()) {
        // renderer option to replace noise with constant value. This can be
        // useful as a profiling aid, to see how much it speeds up to have
        // trivial expense for noise calls.
        if (name == Strings::uperlin || name == Strings::noise ||
            name == Strings::usimplexnoise || name == Strings::usimplex ||
            name == Strings::cell || name == Strings::cellnoise ||
            name == Strings::pcellnoise || name == Strings::pnoise)
            name = ustring("unullnoise");
        else
            name = ustring("nullnoise");
        pass_name = false;
        periodic = false;
        pass_sg = false;
        pass_options = false;
    }

    llvm::Value *opt = NULL;
    if (pass_options) {
        opt = llvm_gen_noise_options (rop, opnum, arg);
    }

    OSL_DEV_ONLY(std::cout << "llvm_gen_noise function name=" << name << std::endl);

    std::string funcname = "osl_" + name.string() + "_" + warg_typecode(&Result,derivs);
    std::vector<llvm::Value *> args;

//    args.push_back (rop.llvm_void_ptr (Result));

    if (pass_name)
        args.push_back (rop.llvm_load_value(*Name));
    llvm::Value *tmpresult = NULL;


    // triple return, or float return with derivs, passes result pointer
    // Always pass result as we can't return a wide type through C ABI
    //if (outdim == 3 || derivs) {
        if (derivs && !Result.has_derivs()) {
            tmpresult = rop.llvm_load_arg (Result, true, op_is_uniform);
            args.push_back (tmpresult);
        }
        else
            args.push_back (rop.llvm_void_ptr (Result));
    //}
    funcname += warg_typecode(S, derivs);
    args.push_back (rop.llvm_load_arg (*S, derivs, op_is_uniform));
    if (T) {
        funcname += warg_typecode(T, derivs);
        args.push_back (rop.llvm_load_arg (*T, derivs, op_is_uniform));
    }

    if (periodic) {
        funcname += warg_typecode (Sper, false /* no derivs */);
        args.push_back (rop.llvm_load_arg (*Sper, false, op_is_uniform));
        if (Tper) {
            funcname += warg_typecode (Tper, false /* no derivs */);
            args.push_back (rop.llvm_load_arg (*Tper, false, op_is_uniform));
        }
    }

    if (pass_sg)
        args.push_back (rop.sg_void_ptr());
    if (pass_options)
        args.push_back (opt);

#ifdef OSL_DEV
    std::cout << "About to push " << funcname << "\n";
    for (size_t i = 0;  i < args.size();  ++i) {
    	{
    		llvm::raw_os_ostream os_cout(std::cout);
    		args[i]->print(os_cout);
    	}
    	std::cout << "\n";
    }
#endif

    // We always pass the result as a parameter, so no return value to store
    /*llvm::Value *r =*/ rop.ll.call_function (funcname.c_str(),
                                             &args[0], (int)args.size());

#if 0
    if (outdim == 1 && !derivs) {
        // Just plain float (no derivs) returns its value
        rop.llvm_store_value (r, Result);
    } else
#endif
    if (derivs && !Result.has_derivs()) {
        // Function needed to take derivs, but our result doesn't have them.
        // We created a temp, now we need to copy to the real result.

        //tmpresult = rop.llvm_ptr_cast (tmpresult, Result.typespec());
        if (op_is_uniform)
            tmpresult = rop.llvm_ptr_cast (tmpresult, Result.typespec());
        else
            tmpresult = rop.llvm_wide_ptr_cast (tmpresult, Result.typespec());

        for (int c = 0;  c < Result.typespec().aggregate();  ++c) {
            llvm::Value *v = rop.llvm_load_value (tmpresult, Result.typespec(),
                                                  0, NULL, c, TypeDesc::UNKNOWN, op_is_uniform);
            rop.llvm_store_value (v, Result, 0, c);
        }
    } // N.B. other cases already stored their result in the right place

    // Clear derivs if result has them but we couldn't compute them
    if (Result.has_derivs() && !derivs)
        rop.llvm_zero_derivs (Result);

    if (rop.shadingsys().profile() >= 1)
        rop.ll.call_function ("osl_count_noise", rop.sg_void_ptr());

    return true;
}



LLVMGEN (llvm_gen_getattribute)
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
    Opcode &op (rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    DASSERT (nargs >= 3 && nargs <= 5);

    bool array_lookup = rop.opargsym(op,nargs-2)->typespec().is_int();
    bool object_lookup = rop.opargsym(op,2)->typespec().is_string() && nargs >= 4;
    int object_slot = (int)object_lookup;
    int attrib_slot = object_slot + 1;
    int index_slot = array_lookup ? nargs - 2 : 0;

    Symbol& Result      = *rop.opargsym (op, 0);
    Symbol& ObjectName  = *rop.opargsym (op, object_slot); // only valid if object_slot is true
    Symbol& Attribute   = *rop.opargsym (op, attrib_slot);
    Symbol& Index       = *rop.opargsym (op, index_slot);  // only valid if array_lookup is true
    Symbol& Destination = *rop.opargsym (op, nargs-1);
    DASSERT (!Result.typespec().is_closure_based() &&
             !ObjectName.typespec().is_closure_based() &&
             !Attribute.typespec().is_closure_based() &&
             !Index.typespec().is_closure_based() &&
             !Destination.typespec().is_closure_based());


    // Special case for get attributes where the result uniformity can differ
    // from the callback
    bool result_is_uniform = rop.isSymbolUniform(Result);
    bool destination_is_uniform = rop.isSymbolUniform(Destination);
    bool attribute_is_uniform = rop.isSymbolUniform(Attribute);

    ASSERT((!array_lookup || rop.isSymbolUniform(Index)) && "incomplete");
    ASSERT((!object_lookup || rop.isSymbolUniform(ObjectName)) && "incomplete");
//    if (false == rop.isSymbolUniform(Attribute))
//    {
//    	std::cout << "getattribute Varying Attribute :" << Attribute.name().c_str() << std::endl;
//    }

    //ASSERT(rop.isSymbolUniform(Attribute) && "incomplete");


    bool op_is_uniform = rop.getAttributesIsUniform(opnum);

    // We'll pass the destination's attribute type directly to the
    // RenderServices callback so that the renderer can perform any
    // necessary conversions from its internal format to OSL's.
    const TypeDesc* dest_type = &Destination.typespec().simpletype();

    std::vector<llvm::Value *> args;
    if (false == op_is_uniform) {
        ASSERT((!result_is_uniform) && (!destination_is_uniform));

        args.push_back (rop.sg_void_ptr());
        args.push_back (rop.ll.constant ((int)Destination.has_derivs()));
        args.push_back (object_lookup ? rop.llvm_load_value (ObjectName) :
                                        rop.ll.constant (ustring()));
        args.push_back (attribute_is_uniform ? rop.llvm_load_value (Attribute) : rop.llvm_void_ptr(Attribute) );
        args.push_back (rop.ll.constant ((int)array_lookup));
        args.push_back (array_lookup ? rop.llvm_load_value (Index) : rop.ll.constant((int)0)); // Never load a symbol that is invalid
        args.push_back (rop.ll.constant_ptr ((void *) dest_type));
        args.push_back (rop.llvm_void_ptr (Destination));
        args.push_back (rop.ll.mask_as_int(rop.ll.current_mask()));

        const char * func_name = attribute_is_uniform ? "osl_get_attribute_batched"
        		                                      : "osl_get_attribute_w16attr_name_batched";
        llvm::Value *r = rop.ll.call_function (func_name, &args[0], args.size());
        rop.llvm_conversion_store_masked_status(r, Result);
    } else {
        ASSERT((!object_lookup || rop.isSymbolUniform(ObjectName)) && rop.isSymbolUniform(Attribute));

        args.push_back (rop.sg_void_ptr());
        args.push_back (rop.ll.constant ((int)Destination.has_derivs()));
        args.push_back (object_lookup ? rop.llvm_load_value (ObjectName) :
                                        rop.ll.constant (ustring()));
        args.push_back (rop.llvm_load_value (Attribute));
        args.push_back (rop.ll.constant ((int)array_lookup));
        args.push_back (array_lookup ? rop.llvm_load_value (Index) : rop.ll.constant((int)0)); // Never load a symbol that is invalid
        args.push_back (rop.ll.constant_ptr ((void *) dest_type));
        llvm::Value *tempUniformDestination = nullptr;
        if (destination_is_uniform)
        {
            args.push_back (rop.llvm_void_ptr (Destination));
        } else {
            tempUniformDestination = rop.llvm_alloca (Destination.typespec(), Destination.has_derivs(), true /*is_uniform*/);
            args.push_back (rop.ll.void_ptr(tempUniformDestination));
        }
        llvm::Value *r = rop.ll.call_function ("osl_get_attribute_batched_uniform"
                , &args[0], args.size());

        if (!destination_is_uniform)
        {
            rop.llvm_broadcast_uniform_value_at(tempUniformDestination, Destination);
        }

        rop.llvm_conversion_store_uniform_status(r, Result);
    }

    return true;
}



LLVMGEN (llvm_gen_gettextureinfo)
{
	OSL_DEV_ONLY(std::cout << "llvm_gen_gettextureinfo" << std::endl);
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 4);

    Symbol& Result   = *rop.opargsym (op, 0);
    Symbol& Filename = *rop.opargsym (op, 1);
    Symbol& Dataname = *rop.opargsym (op, 2);
    Symbol& Data     = *rop.opargsym (op, 3);

    // make sure query string is uniform
    ASSERT(rop.isSymbolUniform(Dataname));

    DASSERT (!Result.typespec().is_closure_based() &&
             Filename.typespec().is_string() &&
             Dataname.typespec().is_string() &&
             !Data.typespec().is_closure_based() &&
             Result.typespec().is_int());

    const TypeDesc* dest_type = &Data.typespec().simpletype();

    std::vector<llvm::Value *> args;

    args.push_back (rop.sg_void_ptr());

    bool fileNameIsUniform = rop.isSymbolUniform(Filename);
    bool dataIsUniform = rop.isSymbolUniform(Data);
    bool resultIsUniform = rop.isSymbolUniform(Result);
    llvm::Value *r = NULL;
    // file name is uniform, generate scalar version of the function
    if (fileNameIsUniform) {
    	OSL_DEV_ONLY(std::cout << "texture file name is uniform." << std::endl);
        RendererServices::TextureHandle *texture_handle = NULL;
        if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        	OSL_DEV_ONLY(std::cout << "Filename=" << *(ustring *)Filename.data() << std::endl);
            texture_handle = rop.renderer()->get_texture_handle (*(ustring *)Filename.data());
            if (! rop.renderer()->good (texture_handle))
                texture_handle = NULL;
        }

        args.push_back (rop.llvm_load_value (Filename));
        args.push_back (rop.ll.constant_ptr (texture_handle));
        args.push_back (rop.llvm_load_value (Dataname));
        // this is passes a TypeDesc to an LLVM op-code
        args.push_back (rop.ll.constant_ptr ((void *) dest_type));
        // destination
        llvm::Value *tempUniformData = nullptr;
        if (dataIsUniform) {
        	OSL_DEV_ONLY(std::cout << "texture info data is uniform." << std::endl);
            args.push_back (rop.llvm_void_ptr (Data));
        }
        else {
        	OSL_DEV_ONLY(std::cout << "texture info data is varying." << std::endl);
            tempUniformData = rop.llvm_alloca (Data.typespec(), Data.has_derivs(), true /*is_uniform*/);
            args.push_back (rop.ll.void_ptr(tempUniformData));
        }

        r = rop.ll.call_function ("osl_get_textureinfo_batched_uniform", &args[0], args.size());

        if (!dataIsUniform) {
            rop.llvm_broadcast_uniform_value_at(tempUniformData, Data);
        }
        if (!resultIsUniform) {
        	rop.llvm_broadcast_uniform_value(r, Result);
        } else {
        	rop.llvm_store_value (r, Result);
        }
    }
    else {
    	OSL_DEV_ONLY(std::cout << "texture filename is varying, running batched version." << std::endl);

        args.push_back (rop.llvm_void_ptr (Filename));
        args.push_back (rop.llvm_load_value (Dataname));
        // this is passes a TypeDesc to an LLVM op-code
        args.push_back (rop.ll.constant_ptr ((void *) dest_type));
        // destination
        args.push_back (rop.llvm_void_ptr (Data));
        args.push_back (rop.ll.mask_as_int(rop.ll.current_mask()));

        r = rop.ll.call_function ("osl_get_textureinfo_batched", &args[0], args.size());
        rop.llvm_conversion_store_masked_status(r, Result);
    }
    /* Do not leave derivs uninitialized */
    // XXX lfeng: how is this different than get attribute info?
    if (Data.has_derivs())
        rop.llvm_zero_derivs (Data);

    return true;
}

LLVMGEN (llvm_gen_getmessage)
{
    // getmessage() has four "flavors":
    //   * getmessage (attribute_name, value)
    //   * getmessage (attribute_name, value[])
    //   * getmessage (source, attribute_name, value)
    //   * getmessage (source, attribute_name, value[])
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 3 || op.nargs() == 4);
    int has_source = (op.nargs() == 4);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Source = *rop.opargsym (op, 1);
    Symbol& Name   = *rop.opargsym (op, 1+has_source);
    Symbol& Data   = *rop.opargsym (op, 2+has_source);
    DASSERT (Result.typespec().is_int() && Name.typespec().is_string());
    DASSERT (has_source == 0 || Source.typespec().is_string());

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value *args[9];
    args[0] = rop.sg_void_ptr();
    args[1] = has_source ? rop.llvm_load_value(Source)
                         : rop.ll.constant(ustring());
    args[2] = rop.llvm_load_value (Name);

    if (Data.typespec().is_closure_based()) {
    	ASSERT(0 && "incomplete");
        // FIXME: secret handshake for closures ...
        args[3] = rop.ll.constant (TypeDesc(TypeDesc::UNKNOWN,
                                              Data.typespec().arraylength()));
        // We need a void ** here so the function can modify the closure
        args[4] = rop.llvm_void_ptr(Data);
    } else {
        args[3] = rop.ll.constant (Data.typespec().simpletype());
        args[4] = rop.llvm_void_ptr (Data);
    }
    args[5] = rop.ll.constant ((int)Data.has_derivs());

    args[6] = rop.ll.constant(rop.inst()->id());
    args[7] = rop.ll.constant(op.sourcefile());
    args[8] = rop.ll.constant(op.sourceline());

    llvm::Value *r = rop.ll.call_function ("osl_getmessage", args, 9);
    rop.llvm_store_value (r, Result);
    return true;
}



LLVMGEN (llvm_gen_setmessage)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 2);
    Symbol& Name   = *rop.opargsym (op, 0);
    Symbol& Data   = *rop.opargsym (op, 1);
    DASSERT (Name.typespec().is_string());

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Name));
    ASSERT(rop.isSymbolUniform(Data));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value *args[7];
    args[0] = rop.sg_void_ptr();
    args[1] = rop.llvm_load_value (Name);
    if (Data.typespec().is_closure_based()) {
    	ASSERT(0 && "incomplete");
        // FIXME: secret handshake for closures ...
        args[2] = rop.ll.constant (TypeDesc(TypeDesc::UNKNOWN,
                                              Data.typespec().arraylength()));
        // We need a void ** here so the function can modify the closure
        args[3] = rop.llvm_void_ptr(Data);
    } else {
        args[2] = rop.ll.constant (Data.typespec().simpletype());
        args[3] = rop.llvm_void_ptr (Data);
    }

    args[4] = rop.ll.constant(rop.inst()->id());
    args[5] = rop.ll.constant(op.sourcefile());
    args[6] = rop.ll.constant(op.sourceline());

    rop.ll.call_function ("osl_setmessage", args, 7);
    return true;
}



LLVMGEN (llvm_gen_get_simple_SG_field)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 1);

    Symbol& Result = *rop.opargsym (op, 0);
    bool is_uniform;
    int sg_index = rop.ShaderGlobalNameToIndex (op.opname(), is_uniform);
    ASSERT (sg_index >= 0);
    llvm::Value *sg_field = rop.ll.GEP (rop.sg_ptr(), 0, sg_index);
    llvm::Value* r = rop.ll.op_load(sg_field);
    rop.llvm_store_value (r, Result);

    return true;
}



LLVMGEN (llvm_gen_calculatenormal)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 2);

    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& P      = *rop.opargsym (op, 1);

    // TODO: because calculatenormal implicitly uses the flip-handedness
    // of the BatchedShaderGlobals, all of its results must be varying
    // TODO: Update uniform discovery to handle widening results that are
    // implicitly dependent upon varying shader globals
    ASSERT(false == rop.isSymbolUniform(Result));
    ASSERT(false == rop.isSymbolUniform(P));

    DASSERT (Result.typespec().is_triple() && P.typespec().is_triple());
    if (! P.has_derivs()) {
        rop.llvm_assign_zero (Result);
        return true;
    }

    std::vector<llvm::Value *> args;
    args.push_back (rop.llvm_void_ptr (Result));
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_void_ptr (P));
    rop.ll.call_function ("osl_calculatenormal_batched", &args[0], args.size());
    if (Result.has_derivs())
        rop.llvm_zero_derivs (Result);
    return true;
}



LLVMGEN (llvm_gen_area)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 2);

    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& P      = *rop.opargsym (op, 1);

    DASSERT (Result.typespec().is_float() && P.typespec().is_triple());
    if (! P.has_derivs()) {
        rop.llvm_assign_zero (Result);
        return true;
    }

    bool op_is_uniform = rop.isSymbolUniform(Result);

    std::vector<const Symbol *> args;
    args.push_back (&Result);
    args.push_back (&P);
    // TODO: dynamically build width suffix
    rop.llvm_call_function (op_is_uniform ? "osl_area" : "osl_area_w16",
                            &(args[0]), 2,
                            /*deriv_ptrs*/ true,
                            op_is_uniform,
                            false /*functionIsLlvmInlined*/,
                            !op_is_uniform/*ptrToReturnStructIs1stArg*/);

    if (Result.has_derivs())
        rop.llvm_zero_derivs (Result);
    return true;
}



LLVMGEN (llvm_gen_spline)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() >= 4 && op.nargs() <= 5);

    bool has_knot_count = (op.nargs() == 5);
    Symbol& Result   = *rop.opargsym (op, 0);
    Symbol& Spline   = *rop.opargsym (op, 1);
    Symbol& Value    = *rop.opargsym (op, 2);
    Symbol& Knot_count = *rop.opargsym (op, 3); // might alias Knots
    Symbol& Knots    = has_knot_count ? *rop.opargsym (op, 4) :
                                        *rop.opargsym (op, 3);

    DASSERT (!Result.typespec().is_closure_based() &&
             Spline.typespec().is_string()  &&
             Value.typespec().is_float() &&
             !Knots.typespec().is_closure_based() &&
             Knots.typespec().is_array() &&
             (!has_knot_count || (has_knot_count && Knot_count.typespec().is_int())));

    ASSERT(rop.isSymbolUniform(Spline));

    std::string name = Strutil::format("osl_%s_", op.opname().c_str());
    std::vector<llvm::Value *> args;
    // only use derivatives for result if:
    //   result has derivs and (value || knots) have derivs
    bool result_derivs = Result.has_derivs() && (Value.has_derivs() || Knots.has_derivs());

    bool result_is_uniform = rop.isSymbolUniform(Result);
    if (false == result_is_uniform)
    	name += warg_lane_count();
    if (result_derivs)
        name += "d";
    if (Result.typespec().is_float())
        name += "f";
    else if (Result.typespec().is_triple())
        name += "v";

    if (false == rop.isSymbolUniform(Value))
    	name += warg_lane_count();
    if (result_derivs && Value.has_derivs())
        name += "d";
    if (Value.typespec().is_float())
        name += "f";
    else if (Value.typespec().is_triple())
        name += "v";

    if (false == rop.isSymbolUniform(Knots))
    	name += warg_lane_count();
    if (result_derivs && Knots.has_derivs())
        name += "d";
    if (Knots.typespec().simpletype().elementtype() == TypeDesc::FLOAT)
        name += "f";
    else if (Knots.typespec().simpletype().elementtype().aggregate == TypeDesc::VEC3)
        name += "v";
    if (false == result_is_uniform) {
        // for simplicity, always call the masked version
        name += "_masked";
    }

    args.push_back (rop.llvm_void_ptr (Result));
    args.push_back (rop.llvm_load_value (Spline));
    args.push_back (rop.llvm_void_ptr (Value)); // make things easy
    args.push_back (rop.llvm_void_ptr (Knots));
    if (has_knot_count)
        args.push_back (rop.llvm_load_value (Knot_count));
    else
        args.push_back (rop.ll.constant ((int)Knots.typespec().arraylength()));
    args.push_back (rop.ll.constant ((int)Knots.typespec().arraylength()));

    if (false == result_is_uniform) {
        // We always call the masked version, need to pass the mask value
        args.push_back (rop.ll.mask_as_int(rop.ll.current_mask()));
    }

    rop.ll.call_function (name.c_str(), &args[0], args.size());

    if (Result.has_derivs() && !result_derivs)
        rop.llvm_zero_derivs (Result);

    return true;
}



static void
llvm_gen_keyword_fill(BackendLLVMWide &rop, Opcode &op, const ClosureRegistry::ClosureEntry *clentry, ustring clname, llvm::Value *mem_void_ptr, int argsoffset)
{
    DASSERT(((op.nargs() - argsoffset) % 2) == 0);

    int Nattrs = (op.nargs() - argsoffset) / 2;

    for (int attr_i = 0; attr_i < Nattrs; ++attr_i) {
        int argno = attr_i * 2 + argsoffset;
        Symbol &Key     = *rop.opargsym (op, argno);
        Symbol &Value   = *rop.opargsym (op, argno + 1);
        ASSERT(Key.typespec().is_string());
        ASSERT(Key.is_constant());
        ustring *key = (ustring *)Key.data();
        TypeDesc ValueType = Value.typespec().simpletype();

        bool legal = false;
        // Make sure there is some keyword arg that has the name and the type
        for (int t = 0; t < clentry->nkeyword; ++t) {
            const ClosureParam &p = clentry->params[clentry->nformal + t];
            // strcmp might be too much, we could precompute the ustring for the param,
            // but in this part of the code is not a big deal
            if (equivalent(p.type,ValueType) && !strcmp(key->c_str(), p.key)) {
                // store data
                DASSERT(p.offset + p.field_size <= clentry->struct_size);
                llvm::Value* dst = rop.ll.offset_ptr (mem_void_ptr, p.offset);
                llvm::Value* src = rop.llvm_void_ptr (Value);
                rop.ll.op_memcpy (dst, src, (int)p.type.size(),
                                    4 /* use 4 byte alignment for now */);
                legal = true;
                break;
            }
        }
        if (!legal) {
            rop.shadingcontext()->warning("Unsupported closure keyword arg \"%s\" for %s (%s:%d)", key->c_str(), clname.c_str(), op.sourcefile().c_str(), op.sourceline());
        }
    }
}



LLVMGEN (llvm_gen_closure)
{
	ASSERT(0 && "incomplete");
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() >= 2); // at least the result and the ID

    Symbol &Result = *rop.opargsym (op, 0);
    int weighted   = rop.opargsym(op,1)->typespec().is_string() ? 0 : 1;
    Symbol *weight = weighted ? rop.opargsym (op, 1) : NULL;
    Symbol &Id     = *rop.opargsym (op, 1+weighted);
    DASSERT(Result.typespec().is_closure());
    DASSERT(Id.typespec().is_string());
    ustring closure_name = *((ustring *)Id.data());

    const ClosureRegistry::ClosureEntry * clentry = rop.shadingsys().find_closure(closure_name);
    if (!clentry) {
        rop.llvm_gen_error (Strutil::format("Closure '%s' is not supported by the current renderer, called from %s:%d in shader \"%s\", layer %d \"%s\", group \"%s\"",
                                     closure_name, op.sourcefile(), op.sourceline(),
                                     rop.inst()->shadername(), rop.layer(),
                                     rop.inst()->layername(), rop.group().name()));
        return false;
    }

    ASSERT (op.nargs() >= (2 + weighted + clentry->nformal));

    // Call osl_allocate_closure_component(closure, id, size).  It returns
    // the memory for the closure parameter data.
    llvm::Value *render_ptr = rop.ll.constant_ptr(rop.shadingsys().renderer(), rop.ll.type_void_ptr());
    llvm::Value *sg_ptr = rop.sg_void_ptr();
    llvm::Value *id_int = rop.ll.constant(clentry->id);
    llvm::Value *size_int = rop.ll.constant(clentry->struct_size);
    llvm::Value *alloc_args[4] = { sg_ptr, id_int, size_int,
                                   weighted ? rop.llvm_void_ptr(*weight) : NULL };
    llvm::Value *return_ptr = weighted ?
          rop.ll.call_function ("osl_allocate_weighted_closure_component", alloc_args, 4)
        : rop.ll.call_function ("osl_allocate_closure_component", alloc_args, 3);
    llvm::Value *comp_void_ptr = return_ptr;

    // For the weighted closures, we need a surrounding "if" so that it's safe
    // for osl_allocate_weighted_closure_component to return NULL (unless we
    // know for sure that it's constant weighted and that the weight is
    // not zero).
    llvm::BasicBlock *next_block = NULL;
    if (weighted && ! (weight->is_constant() && !rop.is_zero(*weight))) {
        llvm::BasicBlock *notnull_block = rop.ll.new_basic_block ("non_null_closure");
        next_block = rop.ll.new_basic_block ("");
        llvm::Value *cond = rop.ll.op_ne (return_ptr, rop.ll.void_ptr_null());
        rop.ll.op_branch (cond, notnull_block, next_block);
        // new insert point is nonnull_block
    }

    llvm::Value *comp_ptr = rop.ll.ptr_cast(comp_void_ptr, rop.llvm_type_closure_component_ptr());
    // Get the address of the primitive buffer, which is the 2nd field
    llvm::Value *mem_void_ptr = rop.ll.GEP (comp_ptr, 0, 2);
    mem_void_ptr = rop.ll.ptr_cast(mem_void_ptr, rop.ll.type_void_ptr());

    // If the closure has a "prepare" method, call
    // prepare(renderer, id, memptr).  If there is no prepare method, just
    // zero out the closure parameter memory.
    if (clentry->prepare) {
        // Call clentry->prepare(renderservices *, int id, void *mem)
        llvm::Value *funct_ptr = rop.ll.constant_ptr((void *)clentry->prepare, rop.llvm_type_prepare_closure_func());
        llvm::Value *args[3] = {render_ptr, id_int, mem_void_ptr};
        rop.ll.call_function (funct_ptr, args, 3);
    } else {
        rop.ll.op_memset (mem_void_ptr, 0, clentry->struct_size, 4 /*align*/);
    }

    // Here is where we fill the struct using the params
    for (int carg = 0; carg < clentry->nformal; ++carg) {
        const ClosureParam &p = clentry->params[carg];
        if (p.key != NULL) break;
        DASSERT(p.offset + p.field_size <= clentry->struct_size);
        Symbol &sym = *rop.opargsym (op, carg + 2 + weighted);
        TypeDesc t = sym.typespec().simpletype();
        if (!sym.typespec().is_closure_array() && !sym.typespec().is_structure()
            && equivalent(t,p.type)) {
            llvm::Value* dst = rop.ll.offset_ptr (mem_void_ptr, p.offset);
            llvm::Value* src = rop.llvm_void_ptr (sym);
            rop.ll.op_memcpy (dst, src, (int)p.type.size(),
                             4 /* use 4 byte alignment for now */);
        } else {
            rop.shadingcontext()->error ("Incompatible formal argument %d to '%s' closure (%s %s, expected %s). Prototypes don't match renderer registry (%s:%d).",
                                         carg + 1, closure_name,
                                         sym.typespec().c_str(), sym.name(), p.type,
                                         op.sourcefile(), op.sourceline());
        }
    }

    // If the closure has a "setup" method, call
    // setup(render_services, id, mem_ptr).
    if (clentry->setup) {
        // Call clentry->setup(renderservices *, int id, void *mem)
        llvm::Value *funct_ptr = rop.ll.constant_ptr((void *)clentry->setup, rop.llvm_type_setup_closure_func());
        llvm::Value *args[3] = {render_ptr, id_int, mem_void_ptr};
        rop.ll.call_function (funct_ptr, args, 3);
    }

    llvm_gen_keyword_fill(rop, op, clentry, closure_name, mem_void_ptr,
                          2 + weighted + clentry->nformal);

    if (next_block)
        rop.ll.op_branch (next_block);

    // Store result at the end, otherwise Ci = modifier(Ci) won't work
    rop.llvm_store_value (return_ptr, Result, 0, NULL, 0);

    return true;
}



LLVMGEN (llvm_gen_pointcloud_search)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() >= 5);
    Symbol& Result     = *rop.opargsym (op, 0);
    Symbol& Filename   = *rop.opargsym (op, 1);
    Symbol& Center     = *rop.opargsym (op, 2);
    Symbol& Radius     = *rop.opargsym (op, 3);
    Symbol& Max_points = *rop.opargsym (op, 4);

    DASSERT (Result.typespec().is_int() && Filename.typespec().is_string() &&
             Center.typespec().is_triple() && Radius.typespec().is_float() &&
             Max_points.typespec().is_int());

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    std::vector<Symbol *> clear_derivs_of; // arguments whose derivs we need to zero at the end
    int attr_arg_offset = 5; // where the opt attrs begin
    Symbol *Sort = NULL;
    if (op.nargs() > 5 && rop.opargsym(op,5)->typespec().is_int()) {
        Sort = rop.opargsym(op,5);
        ++attr_arg_offset;
    }
    int nattrs = (op.nargs() - attr_arg_offset) / 2;

    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());                // 0 sg
    args.push_back (rop.llvm_load_value (Filename));   // 1 filename
    args.push_back (rop.llvm_void_ptr   (Center));     // 2 center
    args.push_back (rop.llvm_load_value (Radius));     // 3 radius
    args.push_back (rop.llvm_load_value (Max_points)); // 4 max_points
    args.push_back (Sort ? rop.llvm_load_value(*Sort)  // 5 sort
                         : rop.ll.constant(0));
    args.push_back (rop.ll.constant_ptr (NULL));      // 6 indices
    args.push_back (rop.ll.constant_ptr (NULL));      // 7 distances
    args.push_back (rop.ll.constant (0));             // 8 derivs_offset
    args.push_back (NULL);                              // 9 nattrs
    size_t capacity = 0x7FFFFFFF; // Lets put a 32 bit limit
    int extra_attrs = 0; // Extra query attrs to search
    // This loop does three things. 1) Look for the special attributes
    // "distance", "index" and grab the pointer. 2) Compute the minimmum
    // size of the provided output arrays to check against max_points
    // 3) push optional args to the arg list
    for (int i = 0; i < nattrs; ++i) {
        Symbol& Name  = *rop.opargsym (op, attr_arg_offset + i*2);
        Symbol& Value = *rop.opargsym (op, attr_arg_offset + i*2 + 1);

        ASSERT (Name.typespec().is_string());
        TypeDesc simpletype = Value.typespec().simpletype();
        if (Name.is_constant() && *((ustring *)Name.data()) == u_index &&
            simpletype.elementtype() == TypeDesc::INT) {
            args[6] = rop.llvm_void_ptr (Value);
        } else if (Name.is_constant() && *((ustring *)Name.data()) == u_distance &&
                   simpletype.elementtype() == TypeDesc::FLOAT) {
            args[7] = rop.llvm_void_ptr (Value);
            if (Value.has_derivs()) {
                if (Center.has_derivs())
                    // deriv offset is the size of the array
                    args[8] = rop.ll.constant ((int)simpletype.numelements());
                else
                    clear_derivs_of.push_back(&Value);
            }
        } else {
            // It is a regular attribute, push it to the arg list
            args.push_back (rop.llvm_load_value (Name));
            args.push_back (rop.ll.constant (simpletype));
            args.push_back (rop.llvm_void_ptr (Value));
            if (Value.has_derivs())
                clear_derivs_of.push_back(&Value);
            extra_attrs++;
        }
        // minimum capacity of the output arrays
        capacity = std::min (simpletype.numelements(), capacity);
    }

    args[9] = rop.ll.constant (extra_attrs);

    // Compare capacity to the requested number of points. The available
    // space on the arrays is a constant, the requested number of
    // points is not, so runtime check.
    llvm::Value *sizeok = rop.ll.op_ge (rop.ll.constant((int)capacity), args[4]); // max_points

    llvm::BasicBlock* sizeok_block = rop.ll.new_basic_block ("then sizeok");
    llvm::BasicBlock* badsize_block = rop.ll.new_basic_block ("else !sizeok");
    llvm::BasicBlock* after_block = rop.ll.new_basic_block ("after sizeok");
    rop.ll.op_branch (sizeok, sizeok_block, badsize_block);
    // N.B. the op_branch sets sizeok_block as the new insert point

    // non-error code case
    llvm::Value *count = rop.ll.call_function ("osl_pointcloud_search", &args[0], args.size());
    // Clear derivs if necessary
    for (size_t i = 0; i < clear_derivs_of.size(); ++i)
        rop.llvm_zero_derivs (*clear_derivs_of[i], count);
    // Store result
    rop.llvm_store_value (count, Result);
    rop.ll.op_branch (after_block);

    // error code case
    rop.ll.set_insert_point (badsize_block);
    args.clear();
    static ustring errorfmt("Arrays too small for pointcloud lookup at (%s:%d)");
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.ll.constant_ptr ((void *)errorfmt.c_str()));
    args.push_back (rop.ll.constant_ptr ((void *)op.sourcefile().c_str()));
    args.push_back (rop.ll.constant (op.sourceline()));
    rop.ll.call_function ("osl_error", &args[0], args.size());

    rop.ll.op_branch (after_block);
    return true;
}



LLVMGEN (llvm_gen_pointcloud_get)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() >= 6);

    Symbol& Result     = *rop.opargsym (op, 0);
    Symbol& Filename   = *rop.opargsym (op, 1);
    Symbol& Indices    = *rop.opargsym (op, 2);
    Symbol& Count      = *rop.opargsym (op, 3);
    Symbol& Attr_name  = *rop.opargsym (op, 4);
    Symbol& Data       = *rop.opargsym (op, 5);

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value *count = rop.llvm_load_value (Count);

    int capacity = std::min ((int)Data.typespec().simpletype().numelements(), (int)Indices.typespec().simpletype().numelements());
    // Check available space
    llvm::Value *sizeok = rop.ll.op_ge (rop.ll.constant(capacity), count);

    llvm::BasicBlock* sizeok_block = rop.ll.new_basic_block ("then sizeok");
    llvm::BasicBlock* badsize_block = rop.ll.new_basic_block ("else !sizeok");
    llvm::BasicBlock* after_block = rop.ll.new_basic_block ("after sizeok");
    rop.ll.op_branch (sizeok, sizeok_block, badsize_block);
    // N.B. sets insert point to true case

    // non-error code case

    // Convert 32bit indices to 64bit
    std::vector<llvm::Value *> args;
    args.clear();
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (rop.llvm_void_ptr (Indices));
    args.push_back (count);
    args.push_back (rop.llvm_load_value (Attr_name));
    args.push_back (rop.ll.constant (Data.typespec().simpletype()));
    args.push_back (rop.llvm_void_ptr (Data));
    llvm::Value *found = rop.ll.call_function ("osl_pointcloud_get", &args[0], args.size());
    rop.llvm_store_value (found, Result);
    if (Data.has_derivs())
        rop.llvm_zero_derivs (Data, count);
    rop.ll.op_branch (after_block);

    // error code case
    rop.ll.set_insert_point (badsize_block);
    args.clear();
    static ustring errorfmt("Arrays too small for pointcloud attribute get at (%s:%d)");
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.ll.constant_ptr ((void *)errorfmt.c_str()));
    args.push_back (rop.ll.constant_ptr ((void *)op.sourcefile().c_str()));
    args.push_back (rop.ll.constant (op.sourceline()));
    rop.ll.call_function ("osl_error", &args[0], args.size());

    rop.ll.op_branch (after_block);
    return true;
}



LLVMGEN (llvm_gen_pointcloud_write)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() >= 3);
    Symbol& Result   = *rop.opargsym (op, 0);
    Symbol& Filename = *rop.opargsym (op, 1);
    Symbol& Pos      = *rop.opargsym (op, 2);
    DASSERT (Result.typespec().is_int() && Filename.typespec().is_string() &&
             Pos.typespec().is_triple());
    DASSERT ((op.nargs() & 1) && "must have an even number of attribs");

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    int nattrs = (op.nargs() - 3) / 2;

    // Generate local space for the names/types/values arrays
    llvm::Value *names = rop.ll.op_alloca (rop.ll.type_string(), nattrs, "pointcloud_write names");
    llvm::Value *types = rop.ll.op_alloca (rop.ll.type_typedesc(), nattrs, "pointcloud_write types");
    llvm::Value *values = rop.ll.op_alloca (rop.ll.type_void_ptr(), nattrs, "pointcloud_write values");

    // Fill in the arrays with the params, use helper function because
    // it's a pain to offset things into the array ourselves.
    for (int i = 0;  i < nattrs;  ++i) {
        Symbol *namesym = rop.opargsym (op, 3+2*i);
        Symbol *valsym = rop.opargsym (op, 3+2*i+1);
        llvm::Value * args[7] = {
            rop.ll.void_ptr (names),
            rop.ll.void_ptr (types),
            rop.ll.void_ptr (values),
            rop.ll.constant (i),
            rop.llvm_load_value (*namesym),  // name[i]
            rop.ll.constant (valsym->typespec().simpletype()), // type[i]
            rop.llvm_void_ptr (*valsym)  // value[i]
        };
        rop.ll.call_function ("osl_pointcloud_write_helper", &args[0], 7);
    }

    llvm::Value * args[7] = {
        rop.sg_void_ptr(),   // shaderglobals pointer
        rop.llvm_load_value (Filename),  // name
        rop.llvm_void_ptr (Pos),   // position
        rop.ll.constant (nattrs),  // number of attributes
        rop.ll.void_ptr (names),   // attribute names array
        rop.ll.void_ptr (types),   // attribute types array
        rop.ll.void_ptr (values)   // attribute values array
    };
    llvm::Value *ret = rop.ll.call_function ("osl_pointcloud_write", &args[0], 7);
    rop.llvm_store_value (ret, Result);

    return true;
}




LLVMGEN (llvm_gen_dict_find)
{
    // OSL has two variants of this function:
    //     dict_find (string dict, string query)
    //     dict_find (int nodeID, string query)
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 3);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Source = *rop.opargsym (op, 1);
    Symbol& Query  = *rop.opargsym (op, 2);
    DASSERT (Result.typespec().is_int() && Query.typespec().is_string() &&
             (Source.typespec().is_int() || Source.typespec().is_string()));

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    bool sourceint = Source.typespec().is_int();  // is it an int?
    llvm::Value *args[3];
    args[0] = rop.sg_void_ptr();
    args[1] = rop.llvm_load_value(Source);
    args[2] = rop.llvm_load_value (Query);
    const char *func = sourceint ? "osl_dict_find_iis" : "osl_dict_find_iss";
    llvm::Value *ret = rop.ll.call_function (func, &args[0], 3);
    rop.llvm_store_value (ret, Result);
    return true;
}



LLVMGEN (llvm_gen_dict_next)
{
    // dict_net is very straightforward -- just insert sg ptr as first arg
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 2);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& NodeID = *rop.opargsym (op, 1);
    DASSERT (Result.typespec().is_int() && NodeID.typespec().is_int());

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value *ret = rop.ll.call_function ("osl_dict_next",
                                               rop.sg_void_ptr(),
                                               rop.llvm_load_value(NodeID));
    rop.llvm_store_value (ret, Result);
    return true;
}



LLVMGEN (llvm_gen_dict_value)
{
    // int dict_value (int nodeID, string attribname, output TYPE value)
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 4);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& NodeID = *rop.opargsym (op, 1);
    Symbol& Name   = *rop.opargsym (op, 2);
    Symbol& Value  = *rop.opargsym (op, 3);
    DASSERT (Result.typespec().is_int() && NodeID.typespec().is_int() &&
             Name.typespec().is_string());
    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value *args[5];
    // arg 0: shaderglobals ptr
    args[0] = rop.sg_void_ptr();
    // arg 1: nodeID
    args[1] = rop.llvm_load_value(NodeID);
    // arg 2: attribute name
    args[2] = rop.llvm_load_value(Name);
    // arg 3: encoded type of Value
    args[3] = rop.ll.constant(Value.typespec().simpletype());
    // arg 4: pointer to Value
    args[4] = rop.llvm_void_ptr (Value);
    llvm::Value *ret = rop.ll.call_function ("osl_dict_value", &args[0], 5);
    rop.llvm_store_value (ret, Result);
    return true;
}



LLVMGEN (llvm_gen_split)
{
    // int split (string str, output string result[], string sep, int maxsplit)
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() >= 3 && op.nargs() <= 5);
    Symbol& R       = *rop.opargsym (op, 0);
    Symbol& Str     = *rop.opargsym (op, 1);
    Symbol& Results = *rop.opargsym (op, 2);
    DASSERT (R.typespec().is_int() && Str.typespec().is_string() &&
             Results.typespec().is_array() &&
             Results.typespec().is_string_based());

    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(R));

    llvm::Value *args[5];
    args[0] = rop.llvm_load_value (Str);
    args[1] = rop.llvm_void_ptr (Results);
    if (op.nargs() >= 4) {
        Symbol& Sep = *rop.opargsym (op, 3);
        DASSERT (Sep.typespec().is_string());
        args[2] = rop.llvm_load_value (Sep);
    } else {
        args[2] = rop.ll.constant ("");
    }
    if (op.nargs() >= 5) {
        Symbol& Maxsplit = *rop.opargsym (op, 4);
        DASSERT (Maxsplit.typespec().is_int());
        args[3] = rop.llvm_load_value (Maxsplit);
    } else {
        args[3] = rop.ll.constant (Results.typespec().arraylength());
    }
    args[4] = rop.ll.constant (Results.typespec().arraylength());
    llvm::Value *ret = rop.ll.call_function ("osl_split", &args[0], 5);
    rop.llvm_store_value (ret, R);
    return true;
}



LLVMGEN (llvm_gen_raytype)
{
    // int raytype (string name)
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 2);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Name = *rop.opargsym (op, 1);
    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));

    llvm::Value *args[2] = { rop.sg_void_ptr(), NULL };
    const char *func = NULL;
    if (Name.is_constant()) {
        // We can statically determine the bit pattern
        ustring name = ((ustring *)Name.data())[0];
        args[1] = rop.ll.constant (rop.shadingsys().raytype_bit (name));
        func = "osl_raytype_bit_batched";
    } else {
        // No way to know which name is being asked for
        args[1] = rop.llvm_get_pointer (Name);
        func = "osl_raytype_name_batched";
    }
    llvm::Value *ret = rop.ll.call_function (func, args, 2);
    rop.llvm_store_value (ret, Result);
    return true;
}



// color blackbody (float temperatureK)
// color wavelength_color (float wavelength_nm)  // same function signature
LLVMGEN (llvm_gen_blackbody)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 2);
    Symbol &Result (*rop.opargsym (op, 0));
    Symbol &Temperature (*rop.opargsym (op, 1));
    ASSERT (Result.typespec().is_triple() && Temperature.typespec().is_float());
    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));
    ASSERT(0 && "incomplete"); // needs uniform version accepting ShaderGlobalsBatched

    llvm::Value* args[3] = { rop.sg_void_ptr(), rop.llvm_void_ptr(Result),
                             rop.llvm_load_value(Temperature) };
    rop.ll.call_function (Strutil::format("osl_%s_vf",op.opname().c_str()).c_str(), args, 3);

    // Punt, zero out derivs.
    // FIXME -- only of some day, someone truly needs blackbody() to
    // correctly return derivs with spatially-varying temperature.
    if (Result.has_derivs())
        rop.llvm_zero_derivs (Result);

    return true;
}



// float luminance (color c)
LLVMGEN (llvm_gen_luminance)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 2);
    Symbol &Result (*rop.opargsym (op, 0));
    Symbol &C (*rop.opargsym (op, 1));
    ASSERT (Result.typespec().is_float() && C.typespec().is_triple());

    // luminance = red * luminance_scale.red + green * luminance_scale.green + blue * luminance_scale.blue;

    // Although color systems can be changed via a ShadingSystem attribute,
    // any change of attributes should cause/require a rebuild of the shaders
    // So we will emit the luminance scale as comple time constants
    // and emit the simple math, vs. incur the overhead of a function call

    Color3 luminance_scale = rop.shadingsys().luminance_scale();
    //
    bool result_is_uniform = rop.isSymbolUniform(Result);
    bool op_is_uniform = rop.isSymbolUniform(C);

    llvm::Value *red_scale = op_is_uniform ? rop.ll.constant(luminance_scale[0]) : rop.ll.wide_constant(luminance_scale[0]);
    llvm::Value *green_scale = op_is_uniform ? rop.ll.constant(luminance_scale[1]) : rop.ll.wide_constant(luminance_scale[1]);
    llvm::Value *blue_scale = op_is_uniform ? rop.ll.constant(luminance_scale[2]) : rop.ll.wide_constant(luminance_scale[2]);

    for (int d = 0;  d < 3;  ++d) {  // deriv
        llvm::Value *red = rop.llvm_load_value (C, d, 0, TypeDesc::UNKNOWN, op_is_uniform);
        llvm::Value *green = rop.llvm_load_value (C, d, 1, TypeDesc::UNKNOWN, op_is_uniform);
        llvm::Value *blue = rop.llvm_load_value (C, d, 2, TypeDesc::UNKNOWN, op_is_uniform);

        llvm::Value *scaled_red = rop.ll.op_mul(red_scale, red);
        llvm::Value *scaled_green = rop.ll.op_mul(green_scale, green);
        llvm::Value *scaled_blue = rop.ll.op_mul(blue_scale, blue);

        llvm::Value *result = rop.ll.op_add(rop.ll.op_add(scaled_red,scaled_green),scaled_blue);

        ASSERT(op_is_uniform || !result_is_uniform);
        if (op_is_uniform && !result_is_uniform)
        {
            result = rop.ll.widen_value(result);
        }

        rop.llvm_store_value (result, Result, d);
        if (! Result.has_derivs())  // skip the derivs if we don't need them
            break;
    }



    return true;
}



LLVMGEN (llvm_gen_isconstant)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 2);
    Symbol &Result (*rop.opargsym (op, 0));
    ASSERT (Result.typespec().is_int());
    Symbol &A (*rop.opargsym (op, 1));
    // TODO: Handle non-uniform case below minding mask values
    ASSERT(rop.isSymbolUniform(Result));

    rop.llvm_store_value (rop.ll.constant(A.is_constant() ? 1 : 0), Result);
    return true;
}



LLVMGEN (llvm_gen_functioncall)
{
	//std::cout << "llvm_gen_functioncall" << std::endl;
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 1);

    Symbol &functionNameSymbol(*rop.opargsym (op, 0));
    ASSERT(functionNameSymbol.is_constant());
    ASSERT(functionNameSymbol.typespec().is_string());
    ustring functionName = *(ustring *)functionNameSymbol.data();

    int exit_count_before_functioncall = rop.ll.masked_exit_count();
    rop.ll.push_function_mask(rop.ll.current_mask());
    llvm::BasicBlock* after_block = rop.ll.push_function ();
    unsigned int op_num_function_starts_at = opnum+1;
    unsigned int op_num_function_ends_at = op.jump(0);
    if (rop.ll.debug_is_enabled()) {
       	ustring file_name = rop.inst()->op(op_num_function_starts_at).sourcefile();
       	unsigned int method_line = rop.inst()->op(op_num_function_starts_at).sourceline();
       	rop.ll.debug_push_inlined_function(functionName, file_name, method_line);
    }

    // Generate the code for the body of the function
    rop.build_llvm_code (op_num_function_starts_at, op_num_function_ends_at);
    rop.ll.op_branch (after_block);

    // Continue on with the previous flow
    if (rop.ll.debug_is_enabled()) {
        rop.ll.debug_pop_inlined_function();
    }
    rop.ll.pop_function ();
    rop.ll.pop_function_mask();

	if (rop.ll.masked_exit_count() > exit_count_before_functioncall)
	{
		// At some point one or more calls to exit have been made
		// we need to apply that exit mask the the current function scope's return mask
		rop.ll.apply_exit_to_mask_stack();
	}

    return true;
}


LLVMGEN (llvm_gen_functioncall_nr)
{
    OSL_DEV_ONLY(std::cout << "llvm_gen_functioncall_nr" << std::endl);
    ASSERT(rop.ll.debug_is_enabled()  && "no return version should only exist when debug is enabled");
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 1);

    Symbol &functionNameSymbol(*rop.opargsym (op, 0));
    ASSERT(functionNameSymbol.is_constant());
    ASSERT(functionNameSymbol.typespec().is_string());
    ustring functionName = *(ustring *)functionNameSymbol.data();

    unsigned int op_num_function_starts_at = opnum+1;
    unsigned int op_num_function_ends_at = op.jump(0);
    ASSERT(op.farthest_jump() == op_num_function_ends_at && "As we are not doing any branching, we should ensure that the inlined function truly ends at the farthest jump");
    {
       	ustring file_name = rop.inst()->op(op_num_function_starts_at).sourcefile();
       	unsigned int method_line = rop.inst()->op(op_num_function_starts_at).sourceline();
       	rop.ll.debug_push_inlined_function(functionName, file_name, method_line);
    }

    // Generate the code for the body of the function
    rop.build_llvm_code (op_num_function_starts_at, op_num_function_ends_at);

    // Continue on with the previous flow
    rop.ll.debug_pop_inlined_function();

    return true;
}

LLVMGEN (llvm_gen_return)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 0);

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
	   rop.ll.op_branch (rop.ll.masked_return_block());
    } else {
        if (op.opname() == Strings::op_exit) {
        	OSL_DEV_ONLY(std::cout << " branching to rop.llvm_exit_instance_block()");
            // If it's a real "exit", totally jump out of the shader instance.
            // The exit instance block will be created if it doesn't yet exist.
            rop.ll.op_branch (rop.llvm_exit_instance_block());
        } else {
        	OSL_DEV_ONLY(std::cout << " branching to rop.ll.return_block()");
            // If it's a "return", jump to the exit point of the function.
            rop.ll.op_branch (rop.ll.return_block());
        }
    }
    // Need an unreachable block for any instuctions after the return
    // or exit
    llvm::BasicBlock* next_block = rop.ll.new_basic_block (std::string("after ")+op.opname().c_str());
    rop.ll.set_insert_point (next_block);

    return true;
}



LLVMGEN (llvm_gen_end)
{
    // Dummy routine needed only for the op_descriptor table
    return false;
}



}; // namespace pvt
OSL_NAMESPACE_EXIT
