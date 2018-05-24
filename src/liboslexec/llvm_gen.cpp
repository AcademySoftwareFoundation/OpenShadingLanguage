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

#include <cmath>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"
#include <OSL/genclosure.h>
#include "backendllvm.h"

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
static ustring op_fprintf("fprintf");
static ustring op_ge("ge");
static ustring op_gt("gt");
static ustring op_hashnoise("hashnoise");
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
#define LLVMGEN_ARGS     BackendLLVM &rop, int opnum

/// Macro that defines the full declaration of an LLVM generator.
/// 
#define LLVMGEN(name)  bool name (LLVMGEN_ARGS)

// Forward decl
LLVMGEN (llvm_gen_generic);



void
BackendLLVM::llvm_gen_debug_printf (string_view message)
{
    ustring s = ustring::format ("(%s %s) %s", inst()->shadername(),
                                 inst()->layername(), message);
    ll.call_function ("osl_printf", sg_void_ptr(), ll.constant("%s\n"),
                      ll.constant(s));
}



void
BackendLLVM::llvm_gen_warning (string_view message)
{
    ll.call_function ("osl_warning", sg_void_ptr(), ll.constant("%s\n"),
                      ll.constant(message));
}



void
BackendLLVM::llvm_gen_error (string_view message)
{
    ll.call_function ("osl_error", sg_void_ptr(), ll.constant("%s\n"),
                      ll.constant(message));
}



void
BackendLLVM::llvm_call_layer (int layer, bool unconditional)
{
    // Make code that looks like:
    //     if (! groupdata->run[parentlayer])
    //         parent_layer (sg, groupdata);
    // if it's a conditional call, or
    //     parent_layer (sg, groupdata);
    // if it's run unconditionally.
    // The code in the parent layer itself will set its 'executed' flag.

    llvm::Value *args[2];
    args[0] = sg_ptr ();
    args[1] = groupdata_ptr ();

    ShaderInstance *parent = group()[layer];
    llvm::Value *trueval = ll.constant_bool(true);
    llvm::Value *layerfield = layer_run_ref(layer_remap(layer));
    llvm::BasicBlock *then_block = NULL, *after_block = NULL;
    if (! unconditional) {
        llvm::Value *executed = ll.op_load (layerfield);
        executed = ll.op_ne (executed, trueval);
        then_block = ll.new_basic_block ("");
        after_block = ll.new_basic_block ("");
        ll.op_branch (executed, then_block, after_block);
        // insert point is now then_block
    }

    // Mark the call as a fast call
    llvm::Value *funccall = ll.call_function (layer_function_name(group(), *parent).c_str(), args, 2);
    if (!parent->entry_layer())
        ll.mark_fast_func_call (funccall);

    if (! unconditional)
        ll.op_branch (after_block);  // also moves insert point
}



void
BackendLLVM::llvm_run_connected_layers (Symbol &sym, int symindex,
                                             int opnum,
                                             std::set<int> *already_run)
{
    if (sym.valuesource() != Symbol::ConnectedVal)
        return;  // Nothing to do

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
            rop.llvm_assign_initial_value (sym);
        }
    }
    return true;
}



// Used for printf, error, warning, format, fprintf
LLVMGEN (llvm_gen_printf)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    // Prepare the args for the call
    
    // Which argument is the format string?  Usually 0, but for op
    // format() and fprintf(), the formatting string is argument #1.
    int format_arg = (op.opname() == "format" || op.opname() == "fprintf") ? 1 : 0;
    Symbol& format_sym = *rop.opargsym (op, format_arg);

    std::vector<llvm::Value*> call_args;
    if (!format_sym.is_constant()) {
        rop.shadingcontext()->warning ("%s must currently have constant format\n",
                                  op.opname().c_str());
        return false;
    }

    // For some ops, we push the shader globals pointer
    if (op.opname() == op_printf || op.opname() == op_error ||
            op.opname() == op_warning || op.opname() == op_fprintf)
        call_args.push_back (rop.sg_void_ptr());

    // fprintf also needs the filename
    if (op.opname() == op_fprintf) {
        Symbol& Filename = *rop.opargsym (op, 0);
        llvm::Value* fn = rop.llvm_load_value (Filename);
        call_args.push_back (fn);
    }

    // We're going to need to adjust the format string as we go, but I'd
    // like to reserve a spot for the char*.
    size_t new_format_slot = call_args.size();
    call_args.push_back(NULL);

    ustring format_ustring = *((ustring*)format_sym.data());
    const char* format = format_ustring.c_str();
    std::string s;
    int arg = format_arg + 1;
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
            ASSERT (! sym.typespec().is_structure_based());

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
                    v = rop.ll.call_function ("osl_closure_to_string", rop.sg_void_ptr(), v);
                    call_args.push_back (v);
                    continue;
                }

                for (int c = 0; c < num_components; c++) {
                    if (c != 0 || a != 0)
                        s += " ";
                    s += ourformat;

                    llvm::Value* loaded = nullptr;
                    if (rop.use_optix() && simpletype.basetype == TypeDesc::STRING) {
                        // In the OptiX case, we use the device_string.
                        llvm::Value* device_string = (sym.is_constant())
                            ? rop.getOrAllocateLLVMGlobal (sym)
                            : rop.llvm_load_value (sym, 0, arrind, c);
                        loaded = rop.llvm_load_device_string_char_ptr (device_string);
                    }
                    else {
                        loaded = rop.llvm_load_value (sym, 0, arrind, c);
                    }

                    if (simpletype.basetype == TypeDesc::FLOAT) {
                        // C varargs convention upconverts float->double.
                        loaded = rop.ll.op_float_to_double(loaded);
                    }

                    if (simpletype.basetype == TypeDesc::INT && rop.use_optix()) {
                        // The printf supported by OptiX expects 8-byte arguments,
                        // so promote int to long long
                        loaded = rop.ll.op_int_to_longlong(loaded);
                    }

                    call_args.push_back (loaded);
                }
            }
            ++arg;
        } else {
            // Everything else -- just copy the character and advance
            s += *format++;
        }
    }

    if (rop.use_optix() && arg > (format_arg + 2)) {
        ASSERT (0 && "OptiX printf only supports 0 or 1 arguments at this time");
    }

    // In OptiX, printf currently supports 0 or 1 arguments, and the signature
    // requires 1 argument, so push a null pointer onto the call args if there
    // is no argument.
    if (rop.use_optix() && arg == format_arg + 1) {
        call_args.push_back(rop.ll.void_ptr_null());
    }

    // Some ops prepend things
    if (op.opname() == op_error || op.opname() == op_warning) {
        std::string prefix = Strutil::format ("Shader %s [%s]: ",
                                              op.opname().c_str(),
                                              rop.inst()->shadername().c_str());
        s = prefix + s;
    }

    // Now go back and put the new format string in its place
    if (! rop.use_optix()) {
        call_args[new_format_slot] = rop.ll.constant (s.c_str());
    }
    else {
        // In the OptiX case, we need to use the pointer to the device_string
        // added to the LLVM Module.
        llvm::Value* device_string = rop.getOrAllocateLLVMGlobal (format_sym);
        call_args[new_format_slot] = rop.llvm_load_device_string_char_ptr (device_string);
    }

    // Construct the function name and call it.
    std::string opname = std::string("osl_") + op.opname().string();
    llvm::Value *ret = rop.ll.call_function (opname.c_str(), &call_args[0],
                                               (int)call_args.size());

    // The format op returns a string value, put in in the right spot
    if (op.opname() == op_format)
        rop.llvm_store_value (ret, *rop.opargsym (op, 0));
    return true;
}



LLVMGEN (llvm_gen_add)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    ASSERT (! A.typespec().is_array() && ! B.typespec().is_array());
    if (Result.typespec().is_closure()) {
        ASSERT (A.typespec().is_closure() && B.typespec().is_closure());
        llvm::Value *valargs[3];
        valargs[0] = rop.sg_void_ptr();
        valargs[1] = rop.llvm_load_value (A);
        valargs[2] = rop.llvm_load_value (B);
        llvm::Value *res = rop.ll.call_function ("osl_add_closure_closure", valargs, 3);
        rop.llvm_store_value (res, Result, 0, NULL, 0);
        return true;
    }

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;

    // The following should handle f+f, v+v, v+f, f+v, i+i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r = rop.ll.op_add (a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1;  d <= 2;  ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *a = rop.loadLLVMValue (A, i, d, type);
                    llvm::Value *b = rop.loadLLVMValue (B, i, d, type);
                    llvm::Value *r = rop.ll.op_add (a, b);
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

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;

    ASSERT (! Result.typespec().is_closure_based() &&
            "subtraction of closures not supported");

    // The following should handle f-f, v-v, v-f, f-v, i-i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r = rop.ll.op_sub (a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1;  d <= 2;  ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *a = rop.loadLLVMValue (A, i, d, type);
                    llvm::Value *b = rop.loadLLVMValue (B, i, d, type);
                    llvm::Value *r = rop.ll.op_sub (a, b);
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

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = !Result.typespec().is_closure_based() && Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    // multiplication involving closures
    if (Result.typespec().is_closure()) {
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
        llvm::Value *res = tfloat ? rop.ll.call_function ("osl_mul_closure_float", valargs, 3)
                                  : rop.ll.call_function ("osl_mul_closure_color", valargs, 3);
        rop.llvm_store_value (res, Result, 0, NULL, 0);
        return true;
    }

    // multiplication involving matrices
    if (Result.typespec().is_matrix()) {
        if (A.typespec().is_float()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_mul_m_ff", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_mul_mf", Result, B, A);
            else ASSERT(0);
        } else if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_mul_mf", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_mul_mm", Result, A, B);
            else ASSERT(0);
        } else ASSERT (0);
        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
        return true;
    }

    // The following should handle f*f, v*v, v*f, f*v, i*i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.llvm_load_value (A, 0, i, type);
        llvm::Value *b = rop.llvm_load_value (B, 0, i, type);
        if (!a || !b)
            return false;
        llvm::Value *r = rop.ll.op_mul (a, b);
        rop.llvm_store_value (r, Result, 0, i);

        if (Result.has_derivs() && (A.has_derivs() || B.has_derivs())) {
            // Multiplication of duals: (a*b, a*b.dx + a.dx*b, a*b.dy + a.dy*b)
            ASSERT (is_float);
            llvm::Value *ax = rop.llvm_load_value (A, 1, i, type);
            llvm::Value *bx = rop.llvm_load_value (B, 1, i, type);
            llvm::Value *abx = rop.ll.op_mul (a, bx);
            llvm::Value *axb = rop.ll.op_mul (ax, b);
            llvm::Value *rx = rop.ll.op_add (abx, axb);
            llvm::Value *ay = rop.llvm_load_value (A, 2, i, type);
            llvm::Value *by = rop.llvm_load_value (B, 2, i, type);
            llvm::Value *aby = rop.ll.op_mul (a, by);
            llvm::Value *ayb = rop.ll.op_mul (ay, b);
            llvm::Value *ry = rop.ll.op_add (aby, ayb);
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

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    ASSERT (! Result.typespec().is_closure_based());

    // division involving matrices
    if (Result.typespec().is_matrix()) {
        if (A.typespec().is_float()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_div_m_ff", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_div_fm", Result, A, B);
            else ASSERT (0);
        } else if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_div_mf", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_div_mm", Result, A, B);
            else ASSERT (0);
        } else ASSERT (0);
        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
        return true;
    }

    // The following should handle f/f, v/v, v/f, f/v, i/i
    // That's all that should be allowed by oslc.
    const char *safe_div = is_float ? "osl_safe_div_fff" : "osl_safe_div_iii";
    bool deriv = (Result.has_derivs() && (A.has_derivs() || B.has_derivs()));
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.llvm_load_value (A, 0, i, type);
        llvm::Value *b = rop.llvm_load_value (B, 0, i, type);
        if (!a || !b)
            return false;
        llvm::Value *a_div_b;
        if (B.is_constant() && ! rop.is_zero(B))
            a_div_b = rop.ll.op_div (a, b);
        else
            a_div_b = rop.ll.call_function (safe_div, a, b);
        llvm::Value *rx = NULL, *ry = NULL;

        if (deriv) {
            // Division of duals: (a/b, 1/b*(ax-a/b*bx), 1/b*(ay-a/b*by))
            ASSERT (is_float);
            llvm::Value *binv;
            if (B.is_constant() && ! rop.is_zero(B))
                binv = rop.ll.op_div (rop.ll.constant(1.0f), b);
            else
                binv = rop.ll.call_function (safe_div, rop.ll.constant(1.0f), b);
            llvm::Value *ax = rop.llvm_load_value (A, 1, i, type);
            llvm::Value *bx = rop.llvm_load_value (B, 1, i, type);
            llvm::Value *a_div_b_mul_bx = rop.ll.op_mul (a_div_b, bx);
            llvm::Value *ax_minus_a_div_b_mul_bx = rop.ll.op_sub (ax, a_div_b_mul_bx);
            rx = rop.ll.op_mul (binv, ax_minus_a_div_b_mul_bx);
            llvm::Value *ay = rop.llvm_load_value (A, 2, i, type);
            llvm::Value *by = rop.llvm_load_value (B, 2, i, type);
            llvm::Value *a_div_b_mul_by = rop.ll.op_mul (a_div_b, by);
            llvm::Value *ay_minus_a_div_b_mul_by = rop.ll.op_sub (ay, a_div_b_mul_by);
            ry = rop.ll.op_mul (binv, ay_minus_a_div_b_mul_by);
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
    int num_components = type.aggregate;

#ifdef OSL_LLVM_NO_BITCODE
    // On Windows 32 bit this calls an unknown instruction, probably need to
    // link with LLVM compiler-rt to fix, for now just fall back to op
    if (is_float)
        return llvm_gen_generic (rop, opnum);
#endif

    // The following should handle f%f, v%v, v%f, i%i
    // That's all that should be allowed by oslc.
    const char *safe_mod = is_float ? "osl_fmod_fff" : "osl_safe_mod_iii";
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r;
        if (B.is_constant() && ! rop.is_zero(B))
            r = rop.ll.op_mod (a, b);
        else
            r = rop.ll.call_function (safe_mod, a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        ASSERT (is_float);
        if (A.has_derivs()) {
            // Modulus of duals: (a mod b, ax, ay)
            for (int d = 1;  d <= 2;  ++d) {
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *deriv = rop.loadLLVMValue (A, i, d, type);
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

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int d = 0;  d < 3;  ++d) {  // dx, dy
        for (int i = 0; i < num_components; i++) {
            llvm::Value *a = rop.llvm_load_value (A, d, i, type);
            llvm::Value *r = rop.ll.op_neg (a);
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

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value *val = rop.llvm_load_value (X, 0, i, type);
        llvm::Value *min = rop.llvm_load_value (Min, 0, i, type);
        llvm::Value *cond = rop.ll.op_lt (val, min);
        val = rop.ll.op_select (cond, min, val);
        llvm::Value *valdx=NULL, *valdy=NULL;
        if (Result.has_derivs()) {
            valdx = rop.llvm_load_value (X, 1, i, type);
            valdy = rop.llvm_load_value (X, 2, i, type);
            llvm::Value *mindx = rop.llvm_load_value (Min, 1, i, type);
            llvm::Value *mindy = rop.llvm_load_value (Min, 2, i, type);
            valdx = rop.ll.op_select (cond, mindx, valdx);
            valdy = rop.ll.op_select (cond, mindy, valdy);
        }
        // Now do the upper bound
        llvm::Value *max = rop.llvm_load_value (Max, 0, i, type);
        cond = rop.ll.op_gt (val, max);
        val = rop.ll.op_select (cond, max, val);
        if (Result.has_derivs()) {
            llvm::Value *maxdx = rop.llvm_load_value (Max, 1, i, type);
            llvm::Value *maxdy = rop.llvm_load_value (Max, 2, i, type);
            valdx = rop.ll.op_select (cond, maxdx, valdx);
            valdy = rop.ll.op_select (cond, maxdy, valdy);
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
    TypeDesc type = Result.typespec().simpletype();
    ASSERT (!Result.typespec().is_closure_based() &&
            Result.typespec().is_floatbased());
    int num_components = type.aggregate;
    int x_components = X.typespec().aggregate();
    bool derivs = (Result.has_derivs() &&
                   (A.has_derivs() || B.has_derivs() || X.has_derivs()));

    llvm::Value *one = rop.ll.constant (1.0f);
    llvm::Value *x = rop.llvm_load_value (X, 0, 0, type);
    llvm::Value *one_minus_x = rop.ll.op_sub (one, x);
    llvm::Value *xx = derivs ? rop.llvm_load_value (X, 1, 0, type) : NULL;
    llvm::Value *xy = derivs ? rop.llvm_load_value (X, 2, 0, type) : NULL;
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.llvm_load_value (A, 0, i, type);
        llvm::Value *b = rop.llvm_load_value (B, 0, i, type);
        if (!a || !b)
            return false;
        if (i > 0 && x_components > 1) {
            // Only need to recompute x and 1-x if they change
            x = rop.llvm_load_value (X, 0, i, type);
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
            llvm::Value *ax = rop.llvm_load_value (A, 1, i, type);
            llvm::Value *bx = rop.llvm_load_value (B, 1, i, type);
            if (i > 0 && x_components > 1)
                xx = rop.llvm_load_value (X, 1, i, type);
            llvm::Value *rx1 = rop.ll.op_mul (a, xx);
            llvm::Value *rx2 = rop.ll.op_mul (ax, one_minus_x);
            llvm::Value *rx = rop.ll.op_sub (rx2, rx1);
            llvm::Value *rx3 = rop.ll.op_mul (b, xx);
            rx = rop.ll.op_add (rx, rx3);
            llvm::Value *rx4 = rop.ll.op_mul (bx, x);
            rx = rop.ll.op_add (rx, rx4);

            llvm::Value *ay = rop.llvm_load_value (A, 2, i, type);
            llvm::Value *by = rop.llvm_load_value (B, 2, i, type);
            if (i > 0 && x_components > 1)
                xy = rop.llvm_load_value (X, 2, i, type);
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

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value *x_val = rop.llvm_load_value (x, 0, i, type);
        llvm::Value *y_val = rop.llvm_load_value (y, 0, i, type);

        llvm::Value* cond = NULL;
        // NOTE(boulos): Using <= instead of < to match old behavior
        // (only matters for derivs)
        if (op.opname() == op_min) {
            cond = rop.ll.op_le (x_val, y_val);
        } else {
            cond = rop.ll.op_gt (x_val, y_val);
        }

        llvm::Value* res_val = rop.ll.op_select (cond, x_val, y_val);
        rop.llvm_store_value (res_val, Result, 0, i);
        if (Result.has_derivs()) {
          llvm::Value* x_dx = rop.llvm_load_value (x, 1, i, type);
          llvm::Value* x_dy = rop.llvm_load_value (x, 2, i, type);
          llvm::Value* y_dx = rop.llvm_load_value (y, 1, i, type);
          llvm::Value* y_dy = rop.llvm_load_value (y, 2, i, type);
          rop.llvm_store_value (rop.ll.op_select(cond, x_dx, y_dx), Result, 1, i);
          rop.llvm_store_value (rop.ll.op_select(cond, x_dy, y_dy), Result, 2, i);
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

    llvm::Value *c = rop.llvm_load_value(Index);
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
            c = rop.ll.call_function ("osl_range_check", args);
            ASSERT (c);
        }
    }

    for (int d = 0;  d < 3;  ++d) {  // deriv
        llvm::Value *val = NULL;
        if (Index.is_constant()) {
            int i = *(int*)Index.data();
            i = Imath::clamp (i, 0, 2);
            val = rop.llvm_load_value (Val, d, i);
        } else {
            val = rop.llvm_load_component_value (Val, d, c);
        }
        rop.llvm_store_value (val, Result, d);
        if (! Result.has_derivs())  // skip the derivs if we don't need them
            break;
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

    llvm::Value *c = rop.llvm_load_value(Index);
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
            c = rop.ll.call_function ("osl_range_check", args);
        }
    }

    for (int d = 0;  d < 3;  ++d) {  // deriv
        llvm::Value *val = rop.llvm_load_value (Val, d, 0, TypeDesc::TypeFloat);
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

    llvm::Value *row = rop.llvm_load_value (Row);
    llvm::Value *col = rop.llvm_load_value (Col);
    if (rop.shadingsys().range_checking()) {
        llvm::Value *args[] = { row, rop.ll.constant(4),
                                rop.ll.constant(M.name()),
                                rop.sg_void_ptr(),
                                rop.ll.constant(op.sourcefile()),
                                rop.ll.constant(op.sourceline()),
                                rop.ll.constant(rop.group().name()),
                                rop.ll.constant(rop.layer()),
                                rop.ll.constant(rop.inst()->layername()),
                                rop.ll.constant(rop.inst()->shadername()) };
        row = rop.ll.call_function ("osl_range_check", args);
        args[0] = col;
        col = rop.ll.call_function ("osl_range_check", args);
    }

    llvm::Value *val = NULL; 
    if (Row.is_constant() && Col.is_constant()) {
        int r = Imath::clamp (((int*)Row.data())[0], 0, 3);
        int c = Imath::clamp (((int*)Col.data())[0], 0, 3);
        int comp = 4 * r + c;
        val = rop.llvm_load_value (M, 0, comp);
    } else {
        llvm::Value *comp = rop.ll.op_mul (row, rop.ll.constant(4));
        comp = rop.ll.op_add (comp, col);
        val = rop.llvm_load_component_value (M, 0, comp);
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

    llvm::Value *row = rop.llvm_load_value (Row);
    llvm::Value *col = rop.llvm_load_value (Col);
    if (rop.shadingsys().range_checking()) {
        llvm::Value *args[] = { row, rop.ll.constant(4),
                                rop.ll.constant(Result.name()),
                                rop.sg_void_ptr(),
                                rop.ll.constant(op.sourcefile()),
                                rop.ll.constant(op.sourceline()),
                                rop.ll.constant(rop.group().name()),
                                rop.ll.constant(rop.layer()),
                                rop.ll.constant(rop.inst()->layername()),
                                rop.ll.constant(rop.inst()->shadername()) };
        row = rop.ll.call_function ("osl_range_check", args);
        args[0] = col;
        col = rop.ll.call_function ("osl_range_check", args);
    }

    llvm::Value *val = rop.llvm_load_value (Val, 0, 0, TypeDesc::TypeFloat);

    if (Row.is_constant() && Col.is_constant()) {
        int r = Imath::clamp (((int*)Row.data())[0], 0, 3);
        int c = Imath::clamp (((int*)Col.data())[0], 0, 3);
        int comp = 4 * r + c;
        rop.llvm_store_value (val, Result, 0, comp);
    } else {
        llvm::Value *comp = rop.ll.op_mul (row, rop.ll.constant(4));
        comp = rop.ll.op_add (comp, col);
        rop.llvm_store_component_value (val, Result, 0, comp);
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

    // Get array index we're interested in
    llvm::Value *index = rop.loadLLVMValue (Index);
    if (! index)
        return false;
    if (rop.shadingsys().range_checking()) {
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
            index = rop.ll.call_function ("osl_range_check", args);
        }
    }

    int num_components = Src.typespec().simpletype().aggregate;
    for (int d = 0;  d <= 2;  ++d) {
        for (int c = 0;  c < num_components;  ++c) {
            llvm::Value *val = rop.llvm_load_value (Src, d, index, c);
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

    // Get array index we're interested in
    llvm::Value *index = rop.loadLLVMValue (Index);
    if (! index)
        return false;
    if (rop.shadingsys().range_checking()) {
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
            index = rop.ll.call_function ("osl_range_check", args);
        }
    }

    int num_components = Result.typespec().simpletype().aggregate;

    // Allow float <=> int casting
    TypeDesc cast;
    if (num_components == 1 && !Result.typespec().is_closure() && !Src.typespec().is_closure() &&
        (Result.typespec().is_int_based() ||  Result.typespec().is_float_based()) &&
        (Src.typespec().is_int_based() ||  Src.typespec().is_float_based())) {
        cast = Result.typespec().simpletype();
        cast.arraylen = 0;
    } else {
        // Try to warn before llvm_fatal_error is called which provides little
        // context as to what went wrong.
        ASSERT (Result.typespec().simpletype().basetype ==
                Src.typespec().simpletype().basetype);
    }

    for (int d = 0;  d <= 2;  ++d) {
        for (int c = 0;  c < num_components;  ++c) {
            llvm::Value *val = rop.loadLLVMValue (Src, c, d, cast);
            rop.llvm_store_value (val, Result, d, index, c);
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

    // First, copy the floats into the vector
    int dmax = Result.has_derivs() ? 3 : 1;
    for (int d = 0;  d < dmax;  ++d) {  // loop over derivs
        for (int c = 0;  c < 3;  ++c) {  // loop over components
            const Symbol& comp = *rop.opargsym (op, c+1+using_space);
            llvm::Value* val = rop.llvm_load_value (comp, d, NULL, 0, TypeDesc::TypeFloat);
            rop.llvm_store_value (val, Result, d, NULL, c);
        }
    }

    // Do the color space conversion in-place, if called for
    if (using_space) {
        llvm::Value *args[3];
        args[0] = rop.sg_void_ptr ();  // shader globals
        args[1] = rop.llvm_void_ptr (Result, 0);  // color
        args[2] = rop.llvm_load_value (Space); // from
        rop.ll.call_function ("osl_prepend_color_from", args, 3);
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

    // First, copy the floats into the vector
    int dmax = Result.has_derivs() ? 3 : 1;
    for (int d = 0;  d < dmax;  ++d) {  // loop over derivs
        for (int c = 0;  c < 3;  ++c) {  // loop over components
            const Symbol& comp = *rop.opargsym (op, c+1+using_space);
            llvm::Value* val = rop.llvm_load_value (comp, d, NULL, 0, TypeDesc::TypeFloat);
            rop.llvm_store_value (val, Result, d, NULL, c);
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
        if (op.opname() == "vector")
            vectype = TypeDesc::VECTOR;
        else if (op.opname() == "normal")
            vectype = TypeDesc::NORMAL;
        llvm::Value *args[8] = { rop.sg_void_ptr(),
            rop.llvm_void_ptr(Result), rop.ll.constant(Result.has_derivs()),
            rop.llvm_void_ptr(Result), rop.ll.constant(Result.has_derivs()),
            rop.llvm_load_value(Space), rop.ll.constant(Strings::common),
            rop.ll.constant((int)vectype) };
        RendererServices *rend (rop.shadingsys().renderer());
        if (rend->transform_points (NULL, from, to, 0.0f, NULL, NULL, 0, vectype)) {
            // renderer potentially knows about a nonlinear transformation.
            // Note that for the case of non-constant strings, passing empty
            // from & to will make transform_points just tell us if ANY 
            // nonlinear transformations potentially are supported.
            rop.ll.call_function ("osl_transform_triple_nonlinear", args, 8);
        } else {
            // definitely not a nonlinear transformation
            rop.ll.call_function ("osl_transform_triple", args, 8);
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

    if (using_two_spaces) {
        llvm::Value *args[4];
        args[0] = rop.sg_void_ptr();  // shader globals
        args[1] = rop.llvm_void_ptr(Result);  // result
        args[2] = rop.llvm_load_value(*rop.opargsym (op, 1));  // from
        args[3] = rop.llvm_load_value(*rop.opargsym (op, 2));  // to
        rop.ll.call_function ("osl_get_from_to_matrix", args, 4);
    } else {
        if (nfloats == 1) {
            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val = ((i%4) == (i/4)) 
                    ? rop.llvm_load_value (*rop.opargsym(op,1+using_space))
                    : rop.ll.constant(0.0f);
                rop.llvm_store_value (src_val, Result, 0, i);
            }
        } else if (nfloats == 16) {
            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val = rop.llvm_load_value (*rop.opargsym(op,i+1+using_space));
                rop.llvm_store_value (src_val, Result, 0, i);
            }
        } else {
            ASSERT (0);
        }
        if (using_space) {
            llvm::Value *args[3];
            args[0] = rop.sg_void_ptr();  // shader globals
            args[1] = rop.llvm_void_ptr(Result);  // result
            args[2] = rop.llvm_load_value(*rop.opargsym (op, 1));  // from
            rop.ll.call_function ("osl_prepend_matrix_from", args, 3);
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

    llvm::Value *args[4];
    args[0] = rop.sg_void_ptr();  // shader globals
    args[1] = rop.llvm_void_ptr(M);  // matrix result
    args[2] = rop.llvm_load_value(From);
    args[3] = rop.llvm_load_value(To);
    llvm::Value *result = rop.ll.call_function ("osl_get_from_to_matrix", args, 4);
    rop.llvm_store_value (result, Result);
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

    if (To->typespec().is_matrix()) {
        // llvm_ops has the matrix version already implemented
        llvm_gen_generic (rop, opnum);
        return true;
    }

    // Named space versions from here on out.
    ustring from, to;  // N.B.: initialize to empty strings
    if ((From == NULL || From->is_constant()) && To->is_constant()) {
        // We can know all the space names at this time
        from = From ? *((ustring *)From->data()) : Strings::common;
        to = *((ustring *)To->data());
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
    TypeDesc::VECSEMANTICS vectype = TypeDesc::POINT;
    if (op.opname() == "transformv")
        vectype = TypeDesc::VECTOR;
    else if (op.opname() == "transformn")
        vectype = TypeDesc::NORMAL;
    llvm::Value *args[8] = { rop.sg_void_ptr(),
        rop.llvm_void_ptr(*P), rop.ll.constant(P->has_derivs()),
        rop.llvm_void_ptr(*Result), rop.ll.constant(Result->has_derivs()),
        rop.llvm_load_value(*From), rop.llvm_load_value(*To),
        rop.ll.constant((int)vectype) };
    RendererServices *rend (rop.shadingsys().renderer());
    if (rend->transform_points (NULL, from, to, 0.0f, NULL, NULL, 0, vectype)) {
        // renderer potentially knows about a nonlinear transformation.
        // Note that for the case of non-constant strings, passing empty
        // from & to will make transform_points just tell us if ANY 
        // nonlinear transformations potentially are supported.
        rop.ll.call_function ("osl_transform_triple_nonlinear", args, 8);
    } else {
        // definitely not a nonlinear transformation
        rop.ll.call_function ("osl_transform_triple", args, 8);
    }
    return true;
}



// transformc (string fromspace, string tospace, color p)
LLVMGEN (llvm_gen_transformc)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 4);
    Symbol *Result = rop.opargsym (op, 0);
    Symbol *From = rop.opargsym (op, 1);
    Symbol *To = rop.opargsym (op, 2);
    Symbol *C = rop.opargsym (op, 3);

    llvm::Value *args[] = { rop.sg_void_ptr(),
        rop.llvm_void_ptr(*C), rop.ll.constant(C->has_derivs()),
        rop.llvm_void_ptr(*Result), rop.ll.constant(Result->has_derivs()),
        rop.llvm_load_value(*From), rop.llvm_load_value(*To) };
    rop.ll.call_function ("osl_transformc", args);
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
    if (Src.has_derivs()) {
        if (Src.typespec().is_float()) {
            llvm::Value *r = rop.ll.call_function ("osl_filterwidth_fdf",
                                                     rop.llvm_void_ptr (Src));
            rop.llvm_store_value (r, Result);
        } else {
            rop.ll.call_function ("osl_filterwidth_vdv",
                                    rop.llvm_void_ptr (Result),
                                    rop.llvm_void_ptr (Src));
        }
        // Don't have 2nd order derivs
        rop.llvm_zero_derivs (Result);
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

    if (A.typespec().is_closure()) {
        ASSERT (B.typespec().is_int() &&
                "Only closure==0 and closure!=0 allowed");
        llvm::Value *a = rop.llvm_load_value (A);
        llvm::Value *b = rop.ll.void_ptr_null ();
        llvm::Value *r = (op.opname()==op_eq) ? rop.ll.op_eq(a,b)
                                              : rop.ll.op_ne(a,b);
        // Convert the single bit bool into an int
        r = rop.ll.op_bool_to_int (r);
        rop.llvm_store_value (r, Result);
        return true;
    }

    int num_components = std::max (A.typespec().aggregate(), B.typespec().aggregate());
    bool float_based = A.typespec().is_floatbased() || B.typespec().is_floatbased();
    TypeDesc cast (float_based ? TypeDesc::FLOAT : TypeDesc::UNKNOWN);

    llvm::Value* final_result = 0;
    ustring opname = op.opname();

    if (rop.use_optix() && A.typespec().is_string()) {
        ASSERT (B.typespec().is_string() && "Only string-to-string comparison is supported");

        // Since we are using device_strings in the OptiX case, we compare the
        // tags rather than comparing the string contents or the pointers
        llvm::Value* string_a = (A.is_constant())
            ? rop.getOrAllocateLLVMGlobal (A)
            : rop.llvm_load_value (A, 0, nullptr, 0);

        llvm::Value* string_b = (B.is_constant())
            ? rop.getOrAllocateLLVMGlobal (B)
            : rop.llvm_load_value (B, 0, nullptr, 0);

        llvm::Value* a = rop.llvm_load_device_string_tag (string_a);
        llvm::Value* b = rop.llvm_load_device_string_tag (string_b);

        if (opname == op_eq) {
            final_result = rop.ll.op_eq (a, b);
        } else if (opname == op_neq) {
            final_result = rop.ll.op_ne (a, b);
        } else {
            // Don't know how to handle this.
            ASSERT (0 && "OptiX only supports equality testing for strings");
        }
        ASSERT (final_result);

        final_result = rop.ll.op_bool_to_int (final_result);
        rop.storeLLVMValue (final_result, Result, 0, 0);
        return true;
    }

    for (int i = 0; i < num_components; i++) {
        // Get A&B component i -- note that these correctly handle mixed
        // scalar/triple comparisons as well as int->float casts as needed.
        llvm::Value* a = rop.loadLLVMValue (A, i, 0, cast);
        llvm::Value* b = rop.loadLLVMValue (B, i, 0, cast);

        // Trickery for mixed matrix/scalar comparisons -- compare
        // on-diagonal to the scalar, off-diagonal to zero
        if (A.typespec().is_matrix() && !B.typespec().is_matrix()) {
            if ((i/4) != (i%4))
                b = rop.ll.constant (0.0f);
        }
        if (! A.typespec().is_matrix() && B.typespec().is_matrix()) {
            if ((i/4) != (i%4))
                a = rop.ll.constant (0.0f);
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

    // Convert the single bit bool into an int for now.
    final_result = rop.ll.op_bool_to_int (final_result);
    rop.storeLLVMValue (final_result, Result, 0, 0);
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

    if (! Result.has_derivs() || ! any_deriv_args) {
        // Don't compute derivs -- either not needed or not provided in args
        if (Result.typespec().aggregate() == TypeDesc::SCALAR) {
            llvm::Value *r = rop.llvm_call_function (name.c_str(),
                                                     &(args[1]), op.nargs()-1);
            rop.llvm_store_value (r, Result);
        } else {
            rop.llvm_call_function (name.c_str(),
                                    (args.size())? &(args[0]): NULL, op.nargs());
        }
        rop.llvm_zero_derivs (Result);
    } else {
        // Cases with derivs
        ASSERT (Result.has_derivs() && any_deriv_args);
        rop.llvm_call_function (name.c_str(),
                                (args.size())? &(args[0]): NULL, op.nargs(),
                                true);
    }
    return true;
}



LLVMGEN (llvm_gen_sincos)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Theta   = *rop.opargsym (op, 0);
    Symbol& Sin_out = *rop.opargsym (op, 1);
    Symbol& Cos_out = *rop.opargsym (op, 2);
    std::vector<llvm::Value *> valargs;
    bool theta_deriv   = Theta.has_derivs();
    bool result_derivs = (Sin_out.has_derivs() || Cos_out.has_derivs());

    std::string name = std::string("osl_sincos_");
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        if (s->has_derivs() && result_derivs  && theta_deriv)
            name += "d";
        if (s->typespec().is_float())
            name += "f";
        else if (s->typespec().is_triple())
            name += "v";
        else ASSERT (0);
    }
    // push back llvm arguments
    valargs.push_back ( (theta_deriv && result_derivs) || Theta.typespec().is_triple() ? 
          rop.llvm_void_ptr (Theta) : rop.llvm_load_value (Theta));
    valargs.push_back (rop.llvm_void_ptr (Sin_out));
    valargs.push_back (rop.llvm_void_ptr (Cos_out));

    rop.ll.call_function (name.c_str(), &valargs[0], 3);

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
    llvm::Value* i32_res = rop.ll.op_bool_to_int(i1_res);
    rop.llvm_store_value(i32_res, result, 0, 0);
    return true;
}


LLVMGEN (llvm_gen_if)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym (op, 0);

    // Load the condition variable and figure out if it's nonzero
    llvm::Value* cond_val = rop.llvm_test_nonzero (cond);

    // Branch on the condition, to our blocks
    llvm::BasicBlock* then_block = rop.ll.new_basic_block ("then");
    llvm::BasicBlock* else_block = rop.ll.new_basic_block ("else");
    llvm::BasicBlock* after_block = rop.ll.new_basic_block ("");
    rop.ll.op_branch (cond_val, then_block, else_block);

    // Then block
    rop.build_llvm_code (opnum+1, op.jump(0), then_block);
    rop.ll.op_branch (after_block);

    // Else block
    rop.build_llvm_code (op.jump(0), op.jump(1), else_block);
    rop.ll.op_branch (after_block);  // insert point is now after_block

    // Continue on with the previous flow
    return true;
}



LLVMGEN (llvm_gen_loop_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym (op, 0);

    // Branch on the condition, to our blocks
    llvm::BasicBlock* cond_block = rop.ll.new_basic_block ("cond");
    llvm::BasicBlock* body_block = rop.ll.new_basic_block ("body");
    llvm::BasicBlock* step_block = rop.ll.new_basic_block ("step");
    llvm::BasicBlock* after_block = rop.ll.new_basic_block ("");
    // Save the step and after block pointers for possible break/continue
    rop.ll.push_loop (step_block, after_block);

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
    rop.ll.pop_loop ();

    return true;
}



LLVMGEN (llvm_gen_loopmod_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 0);
    if (op.opname() == op_break) {
        rop.ll.op_branch (rop.ll.loop_after_block());
    } else {  // continue
        rop.ll.op_branch (rop.ll.loop_step_block());
    }
    llvm::BasicBlock* next_block = rop.ll.new_basic_block ("");
    rop.ll.set_insert_point (next_block);
    return true;
}



static llvm::Value *
llvm_gen_texture_options (BackendLLVM &rop, int opnum,
                          int first_optional_arg, bool tex3d, int nchans,
                          llvm::Value* &alpha, llvm::Value* &dalphadx,
                          llvm::Value* &dalphady, llvm::Value* &errormessage)
{
    llvm::Value* opt = rop.ll.call_function ("osl_get_texture_options",
                                             rop.sg_void_ptr());
    llvm::Value* missingcolor = NULL;
    TextureOpt optdefaults;  // So we can check the defaults
    bool swidth_set = false, twidth_set = false, rwidth_set = false;
    bool sblur_set = false, tblur_set = false, rblur_set = false;
    bool swrap_set = false, twrap_set = false, rwrap_set = false;
    bool firstchannel_set = false, fill_set = false, interp_set = false;
    bool time_set = false, subimage_set = false;

    Opcode &op (rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg;  a < op.nargs();  ++a) {
        Symbol &Name (*rop.opargsym(op,a));
        ASSERT (Name.typespec().is_string() &&
                "optional texture token must be a string");
        ASSERT (a+1 < op.nargs() && "malformed argument list for texture");
        ustring name = *(ustring *)Name.data();
        ++a;  // advance to next argument

        if (! name)    // skip empty string param name
            continue;

        Symbol &Val (*rop.opargsym(op,a));
        TypeDesc valtype = Val.typespec().simpletype ();
        const int *ival = Val.typespec().is_int() && Val.is_constant() ? (const int *)Val.data() : NULL;
        const float *fval = Val.typespec().is_float() && Val.is_constant() ? (const float *)Val.data() : NULL;

#define PARAM_INT(paramname)                                            \
        if (name == Strings::paramname && valtype == TypeDesc::INT)   { \
            if (! paramname##_set &&                                    \
                ival && *ival == optdefaults.paramname)                 \
                continue;     /* default constant */                    \
            llvm::Value *val = rop.llvm_load_value (Val);               \
            rop.ll.call_function ("osl_texture_set_" #paramname, opt, val); \
            paramname##_set = true;                                     \
            continue;                                                   \
        }

#define PARAM_FLOAT(paramname)                                          \
        if (name == Strings::paramname &&                               \
            (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) { \
            if (! paramname##_set &&                                    \
                ((ival && *ival == optdefaults.paramname) ||            \
                 (fval && *fval == optdefaults.paramname)))             \
                continue;     /* default constant */                    \
            llvm::Value *val = rop.llvm_load_value (Val);               \
            if (valtype == TypeDesc::INT)                               \
                val = rop.ll.op_int_to_float (val);                     \
            rop.ll.call_function ("osl_texture_set_" #paramname, opt, val); \
            paramname##_set = true;                                     \
            continue;                                                   \
        }

#define PARAM_FLOAT_STR(paramname)                                      \
        if (name == Strings::paramname &&                               \
            (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) { \
            if (! s##paramname##_set && ! t##paramname##_set &&         \
                ! r##paramname##_set &&                                 \
                ((ival && *ival == optdefaults.s##paramname) ||         \
                 (fval && *fval == optdefaults.s##paramname)))          \
                continue;     /* default constant */                    \
            llvm::Value *val = rop.llvm_load_value (Val);               \
            if (valtype == TypeDesc::INT)                               \
                val = rop.ll.op_int_to_float (val);                     \
            rop.ll.call_function ("osl_texture_set_st" #paramname, opt, val); \
            if (tex3d)                                                  \
                rop.ll.call_function ("osl_texture_set_r" #paramname, opt, val); \
            s##paramname##_set = true;                                  \
            t##paramname##_set = true;                                  \
            r##paramname##_set = true;                                  \
            continue;                                                   \
        }

#define PARAM_STRING_CODE(paramname,decoder,fieldname)                  \
        if (name == Strings::paramname && valtype == TypeDesc::STRING) { \
            if (Val.is_constant()) {                                    \
                int code = decoder (*(ustring *)Val.data());            \
                if (! paramname##_set && code == optdefaults.fieldname) \
                    continue;                                           \
                if (code >= 0) {                                        \
                    llvm::Value *val = rop.ll.constant (code);          \
                    rop.ll.call_function ("osl_texture_set_" #paramname "_code", opt, val); \
                }                                                       \
            } else {                                                    \
                llvm::Value *val = rop.llvm_load_value (Val);           \
                rop.ll.call_function ("osl_texture_set_" #paramname, opt, val); \
            }                                                           \
            paramname##_set = true;                                     \
            continue;                                                   \
        }

        PARAM_FLOAT_STR (width)
        PARAM_FLOAT (swidth)
        PARAM_FLOAT (twidth)
        PARAM_FLOAT (rwidth)
        PARAM_FLOAT_STR (blur)
        PARAM_FLOAT (sblur)
        PARAM_FLOAT (tblur)
        PARAM_FLOAT (rblur)

        if (name == Strings::wrap && valtype == TypeDesc::STRING) {
            if (Val.is_constant()) {
                int mode = TextureOpt::decode_wrapmode (*(ustring *)Val.data());
                llvm::Value *val = rop.ll.constant (mode);
                rop.ll.call_function ("osl_texture_set_stwrap_code", opt, val);
                if (tex3d)
                    rop.ll.call_function ("osl_texture_set_rwrap_code", opt, val);
            } else {
                llvm::Value *val = rop.llvm_load_value (Val);
                rop.ll.call_function ("osl_texture_set_stwrap", opt, val);
                if (tex3d)
                    rop.ll.call_function ("osl_texture_set_rwrap", opt, val);
            }
            swrap_set = twrap_set = rwrap_set = true;
            continue;
        }
        PARAM_STRING_CODE(swrap, TextureOpt::decode_wrapmode, swrap)
        PARAM_STRING_CODE(twrap, TextureOpt::decode_wrapmode, twrap)
        PARAM_STRING_CODE(rwrap, TextureOpt::decode_wrapmode, rwrap)

        PARAM_FLOAT (fill)
        PARAM_FLOAT (time)
        PARAM_INT (firstchannel)
        PARAM_INT (subimage)

        if (name == Strings::subimage && valtype == TypeDesc::STRING) {
            if (Val.is_constant()) {
                ustring v = *(ustring *)Val.data();
                if (! v && ! subimage_set) {
                    continue;     // Ignore nulls unless they are overrides
                }
            }
            llvm::Value *val = rop.llvm_load_value (Val);
            rop.ll.call_function ("osl_texture_set_subimagename", opt, val);
            subimage_set = true;
            continue;
        }

        PARAM_STRING_CODE (interp, tex_interp_to_code, interpmode)

        if (name == Strings::alpha && valtype == TypeDesc::FLOAT) {
            alpha = rop.llvm_get_pointer (Val);
            if (Val.has_derivs()) {
                dalphadx = rop.llvm_get_pointer (Val, 1);
                dalphady = rop.llvm_get_pointer (Val, 2);
                // NO z derivs!  dalphadz = rop.llvm_get_pointer (Val, 3);
            }
            continue;
        }
        if (name == Strings::errormessage && valtype == TypeDesc::STRING) {
            errormessage = rop.llvm_get_pointer (Val);
            continue;
        }
        if (name == Strings::missingcolor &&
                   equivalent(valtype,TypeDesc::TypeColor)) {
            if (! missingcolor) {
                // If not already done, allocate enough storage for the
                // missingcolor value (4 floats), and call the special 
                // function that points the TextureOpt.missingcolor to it.
                missingcolor = rop.ll.op_alloca(rop.ll.type_float(), 4);
                rop.ll.call_function ("osl_texture_set_missingcolor_arena",
                                      opt, rop.ll.void_ptr(missingcolor));
            }
            rop.ll.op_memcpy (rop.ll.void_ptr(missingcolor),
                              rop.llvm_void_ptr(Val), (int)sizeof(Color3));
            continue;
        }
        if (name == Strings::missingalpha && valtype == TypeDesc::FLOAT) {
            if (! missingcolor) {
                // If not already done, allocate enough storage for the
                // missingcolor value (4 floats), and call the special 
                // function that points the TextureOpt.missingcolor to it.
                missingcolor = rop.ll.op_alloca(rop.ll.type_float(), 4);
                rop.ll.call_function ("osl_texture_set_missingcolor_arena",
                                      opt, missingcolor);
            }
            llvm::Value *val = rop.llvm_load_value (Val);
            rop.ll.call_function ("osl_texture_set_missingcolor_alpha",
                                    opt, rop.ll.constant(nchans), val);
            continue;

        }
        rop.shadingcontext()->error ("Unknown texture%s optional argument: \"%s\", <%s> (%s:%d)",
                                     tex3d ? "3d" : "",
                                     name.c_str(), valtype.c_str(),
                                     op.sourcefile().c_str(), op.sourceline());
#undef PARAM_INT
#undef PARAM_FLOAT
#undef PARAM_FLOAT_STR
#undef PARAM_STRING_CODE

#if 0
        // Helps me find any constant optional params that aren't elided
        if (Name.is_constant() && Val.is_constant()) {
            std::cout << "! texture constant optional arg '" << name << "'\n";
            if (Val.typespec().is_float()) std::cout << "\tf " << *(float *)Val.data() << "\n";
            if (Val.typespec().is_int()) std::cout << "\ti " << *(int *)Val.data() << "\n";
            if (Val.typespec().is_string()) std::cout << "\t" << *(ustring *)Val.data() << "\n";
        }
#endif
    }

    return opt;
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
    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
                                    false /*3d*/, nchans,
                                    alpha, dalphadx, dalphady, errormessage);

    // Now call the osl_texture function, passing the options and all the
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
    args.push_back (rop.llvm_load_value (S));
    args.push_back (rop.llvm_load_value (T));
    if (user_derivs) {
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 4)));
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 5)));
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 6)));
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 7)));
    } else {
        // Auto derivs of S and T
        args.push_back (rop.llvm_load_value (S, 1));
        args.push_back (rop.llvm_load_value (T, 1));
        args.push_back (rop.llvm_load_value (S, 2));
        args.push_back (rop.llvm_load_value (T, 2));
    }
    args.push_back (rop.ll.constant (nchans));
    args.push_back (rop.ll.void_ptr (rop.llvm_get_pointer (Result, 0)));
    args.push_back (rop.ll.void_ptr (rop.llvm_get_pointer (Result, 1)));
    args.push_back (rop.ll.void_ptr (rop.llvm_get_pointer (Result, 2)));
    args.push_back (rop.ll.void_ptr (alpha    ? alpha    : rop.ll.void_ptr_null()));
    args.push_back (rop.ll.void_ptr (dalphadx ? dalphadx : rop.ll.void_ptr_null()));
    args.push_back (rop.ll.void_ptr (dalphady ? dalphady : rop.ll.void_ptr_null()));
    args.push_back (rop.ll.void_ptr (errormessage ? errormessage : rop.ll.void_ptr_null()));
    rop.ll.call_function ("osl_texture", &args[0], (int)args.size());
    rop.generated_texture_call (texture_handle != NULL);
    return true;
}



LLVMGEN (llvm_gen_texture3d)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Filename = *rop.opargsym (op, 1);
    Symbol &P = *rop.opargsym (op, 2);
    int nchans = Result.typespec().aggregate();

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
    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
                                    true /*3d*/, nchans,
                                    alpha, dalphadx, dalphady, errormessage);

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
            llvm::Value *vzero = rop.ll.op_alloca (rop.ll.type_triple());
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
    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
                                    false /*3d*/, nchans,
                                    alpha, dalphadx, dalphady, errormessage);

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
llvm_gen_trace_options (BackendLLVM &rop, int opnum,
                        int first_optional_arg)
{
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



static std::string
arg_typecode (Symbol *sym, bool derivs)
{
    const TypeSpec &t (sym->typespec());
    if (t.is_int())
        return "i";
    else if (t.is_matrix())
        return "m";
    else if (t.is_string())
        return "s";

    std::string name;
    if (derivs)
        name = "d";
    if (t.is_float())
        name += "f";
    else if (t.is_triple())
        name += "v";
    else ASSERT (0);
    return name;
}



static llvm::Value *
llvm_gen_noise_options (BackendLLVM &rop, int opnum,
                        int first_optional_arg)
{
    llvm::Value* opt = rop.ll.call_function ("osl_get_noise_options",
                                             rop.sg_void_ptr());

    // TODO: implement noise options in OptiX
    if (rop.use_optix()) {
        return opt;
    }

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
    int outdim = Result.typespec().is_triple() ? 3 : 1;
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
    } else if (name == Strings::hash || name == Strings::hashnoise) {
        name = periodic ? Strings::phashnoise : Strings::hashnoise;
        derivs = false;  // hash noise derivs are always zero
    } else if (name == Strings::simplex && !periodic) {
        name = Strings::simplexnoise;
    } else if (name == Strings::usimplex && !periodic) {
        name = Strings::usimplexnoise;
    } else if (name == Strings::gabor) {
        // already named
        pass_name = true;
        pass_sg = true;
        pass_options = true;
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
            name == Strings::hash || name == Strings::hashnoise ||
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

    std::string funcname = "osl_" + name.string() + "_" + arg_typecode(&Result,derivs);
    std::vector<llvm::Value *> args;
    if (pass_name) {
        args.push_back (rop.llvm_load_value(*Name));
    }
    llvm::Value *tmpresult = NULL;
    // triple return, or float return with derivs, passes result pointer
    if (outdim == 3 || derivs) {
        if (derivs && !Result.has_derivs()) {
            tmpresult = rop.llvm_load_arg (Result, true);
            args.push_back (tmpresult);
        }
        else
            args.push_back (rop.llvm_void_ptr (Result));
    }
    funcname += arg_typecode(S, derivs);
    args.push_back (rop.llvm_load_arg (*S, derivs));
    if (T) {
        funcname += arg_typecode(T, derivs);
        args.push_back (rop.llvm_load_arg (*T, derivs));
    }

    if (periodic) {
        funcname += arg_typecode (Sper, false /* no derivs */);
        args.push_back (rop.llvm_load_arg (*Sper, false));
        if (Tper) {
            funcname += arg_typecode (Tper, false /* no derivs */);
            args.push_back (rop.llvm_load_arg (*Tper, false));
        }
    }

    if (pass_sg)
        args.push_back (rop.sg_void_ptr());
    if (pass_options)
        args.push_back (opt);

#if 0
    llvm::outs() << "About to push " << funcname << "\n";
    for (size_t i = 0;  i < args.size();  ++i)
        llvm::outs() << "    " << *args[i] << "\n";
#endif

    llvm::Value *r = rop.ll.call_function (funcname.c_str(),
                                             &args[0], (int)args.size());
    if (outdim == 1 && !derivs) {
        // Just plain float (no derivs) returns its value
        rop.llvm_store_value (r, Result);
    } else if (derivs && !Result.has_derivs()) {
        // Function needed to take derivs, but our result doesn't have them.
        // We created a temp, now we need to copy to the real result.
        tmpresult = rop.llvm_ptr_cast (tmpresult, Result.typespec());
        for (int c = 0;  c < Result.typespec().aggregate();  ++c) {
            llvm::Value *v = rop.llvm_load_value (tmpresult, Result.typespec(),
                                                  0, NULL, c);
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

    // We'll pass the destination's attribute type directly to the 
    // RenderServices callback so that the renderer can perform any
    // necessary conversions from its internal format to OSL's.
    const TypeDesc* dest_type = &Destination.typespec().simpletype();

    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.ll.constant ((int)Destination.has_derivs()));
    args.push_back (object_lookup ? rop.llvm_load_value (ObjectName) :
                                    rop.ll.constant (ustring()));
    args.push_back (rop.llvm_load_value (Attribute));
    args.push_back (rop.ll.constant ((int)array_lookup));
    args.push_back (rop.llvm_load_value (Index));
    args.push_back (rop.ll.constant_ptr ((void *) dest_type));
    args.push_back (rop.llvm_void_ptr (Destination));

    llvm::Value *r = rop.ll.call_function ("osl_get_attribute", &args[0], args.size());
    rop.llvm_store_value (r, Result);

    return true;
}



LLVMGEN (llvm_gen_gettextureinfo)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 4);

    Symbol& Result   = *rop.opargsym (op, 0);
    Symbol& Filename = *rop.opargsym (op, 1);
    Symbol& Dataname = *rop.opargsym (op, 2);
    Symbol& Data     = *rop.opargsym (op, 3);

    DASSERT (!Result.typespec().is_closure_based() &&
             Filename.typespec().is_string() && 
             Dataname.typespec().is_string() &&
             !Data.typespec().is_closure_based() && 
             Result.typespec().is_int());

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
    args.push_back (rop.llvm_load_value (Dataname));
    // this is passes a TypeDesc to an LLVM op-code
    args.push_back (rop.ll.constant((int) Data.typespec().simpletype().basetype));
    args.push_back (rop.ll.constant((int) Data.typespec().simpletype().arraylen));
    args.push_back (rop.ll.constant((int) Data.typespec().simpletype().aggregate));
    // destination
    args.push_back (rop.llvm_void_ptr (Data));

    llvm::Value *r = rop.ll.call_function ("osl_get_textureinfo",
                                           &args[0], args.size());
    rop.llvm_store_value (r, Result);
    /* Do not leave derivs uninitialized */
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

    llvm::Value *args[9];
    args[0] = rop.sg_void_ptr();
    args[1] = has_source ? rop.llvm_load_value(Source) 
                         : rop.ll.constant(ustring());
    args[2] = rop.llvm_load_value (Name);

    if (Data.typespec().is_closure_based()) {
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

    llvm::Value *args[7];
    args[0] = rop.sg_void_ptr();
    args[1] = rop.llvm_load_value (Name);
    if (Data.typespec().is_closure_based()) {
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
    int sg_index = rop.ShaderGlobalNameToIndex (op.opname());
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

    DASSERT (Result.typespec().is_triple() && P.typespec().is_triple());
    if (! P.has_derivs()) {
        rop.llvm_assign_zero (Result);
        return true;
    }
    
    std::vector<llvm::Value *> args;
    args.push_back (rop.llvm_void_ptr (Result));
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_void_ptr (P));
    rop.ll.call_function ("osl_calculatenormal", &args[0], args.size());
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
    
    llvm::Value *r = rop.ll.call_function ("osl_area", rop.llvm_void_ptr (P));
    rop.llvm_store_value (r, Result);
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

    std::string name = Strutil::format("osl_%s_", op.opname().c_str());
    std::vector<llvm::Value *> args;
    // only use derivatives for result if:
    //   result has derivs and (value || knots) have derivs
    bool result_derivs = Result.has_derivs() && (Value.has_derivs() || Knots.has_derivs());

    if (result_derivs)
        name += "d";
    if (Result.typespec().is_float())
        name += "f";
    else if (Result.typespec().is_triple())
        name += "v";

    if (result_derivs && Value.has_derivs())
        name += "d";
    if (Value.typespec().is_float())
        name += "f";
    else if (Value.typespec().is_triple())
        name += "v";

    if (result_derivs && Knots.has_derivs())
        name += "d";
    if (Knots.typespec().simpletype().elementtype() == TypeDesc::FLOAT)
        name += "f";
    else if (Knots.typespec().simpletype().elementtype().aggregate == TypeDesc::VEC3)
        name += "v";

    args.push_back (rop.llvm_void_ptr (Result));
    args.push_back (rop.llvm_load_value (Spline));
    args.push_back (rop.llvm_void_ptr (Value)); // make things easy
    args.push_back (rop.llvm_void_ptr (Knots));
    if (has_knot_count)
        args.push_back (rop.llvm_load_value (Knot_count));
    else
        args.push_back (rop.ll.constant ((int)Knots.typespec().arraylength()));
    args.push_back (rop.ll.constant ((int)Knots.typespec().arraylength()));
    rop.ll.call_function (name.c_str(), &args[0], args.size());

    if (Result.has_derivs() && !result_derivs)
        rop.llvm_zero_derivs (Result);

    return true;
}



static void
llvm_gen_keyword_fill(BackendLLVM &rop, Opcode &op, const ClosureRegistry::ClosureEntry *clentry, ustring clname, llvm::Value *mem_void_ptr, int argsoffset)
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

        if (rop.use_optix() && sym.typespec().is_string()) {
            // For OptiX, need to copy the entire 16-byte device_string, which
            // is a struct consisting of the 8-byte tag and the char*.
            llvm::Value* dst = rop.ll.offset_ptr (mem_void_ptr, p.offset);
            llvm::Value* src = nullptr;
            src = (sym.is_constant())
                ? rop.getOrAllocateLLVMGlobal (sym)
                : rop.llvm_load_value (sym, 0, nullptr, 0);
            src = rop.ll.int_to_ptr_cast(src);
            rop.ll.op_memcpy (dst, src, (int)p.type.size(),
                              4 /* use 4 byte alignment for now */);
        }
        else if (!sym.typespec().is_closure_array() && !sym.typespec().is_structure()
                 && equivalent(t,p.type)) {
            llvm::Value* dst = rop.ll.offset_ptr (mem_void_ptr, p.offset);
            llvm::Value* src = nullptr;
            src = rop.llvm_void_ptr (sym);
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

    llvm::BasicBlock* sizeok_block = rop.ll.new_basic_block ("then");
    llvm::BasicBlock* badsize_block = rop.ll.new_basic_block ("else");
    llvm::BasicBlock* after_block = rop.ll.new_basic_block ("");
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

    llvm::Value *count = rop.llvm_load_value (Count);

    int capacity = std::min ((int)Data.typespec().simpletype().numelements(), (int)Indices.typespec().simpletype().numelements());
    // Check available space
    llvm::Value *sizeok = rop.ll.op_ge (rop.ll.constant(capacity), count);

    llvm::BasicBlock* sizeok_block = rop.ll.new_basic_block ("then");
    llvm::BasicBlock* badsize_block = rop.ll.new_basic_block ("else");
    llvm::BasicBlock* after_block = rop.ll.new_basic_block ("");
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

    int nattrs = (op.nargs() - 3) / 2;

    // Generate local space for the names/types/values arrays
    llvm::Value *names = rop.ll.op_alloca (rop.ll.type_string(), nattrs);
    llvm::Value *types = rop.ll.op_alloca (rop.ll.type_typedesc(), nattrs);
    llvm::Value *values = rop.ll.op_alloca (rop.ll.type_void_ptr(), nattrs);

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
    llvm::Value *args[2] = { rop.sg_void_ptr(), NULL };
    const char *func = NULL;
    if (Name.is_constant()) {
        // We can statically determine the bit pattern
        ustring name = ((ustring *)Name.data())[0];
        args[1] = rop.ll.constant (rop.shadingsys().raytype_bit (name));
        func = "osl_raytype_bit";
    } else {
        // No way to know which name is being asked for
        args[1] = rop.llvm_get_pointer (Name);
        func = "osl_raytype_name";
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

    bool deriv = C.has_derivs() && Result.has_derivs();
    llvm::Value* args[3] = { rop.sg_void_ptr(), rop.llvm_void_ptr(Result),
                             rop.llvm_void_ptr(C) };
    rop.ll.call_function (deriv ? "osl_luminance_dfdv" : "osl_luminance_fv",
                            args, 3);

    if (Result.has_derivs() && !C.has_derivs())
        rop.llvm_zero_derivs (Result);

    return true;
}



LLVMGEN (llvm_gen_isconstant)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 2);
    Symbol &Result (*rop.opargsym (op, 0));
    ASSERT (Result.typespec().is_int());
    Symbol &A (*rop.opargsym (op, 1));
    rop.llvm_store_value (rop.ll.constant(A.is_constant() ? 1 : 0), Result);
    return true;
}



LLVMGEN (llvm_gen_functioncall)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 1);

    llvm::BasicBlock* after_block = rop.ll.push_function ();

    // Generate the code for the body of the function
    rop.build_llvm_code (opnum+1, op.jump(0));
    rop.ll.op_branch (after_block);

    // Continue on with the previous flow
    rop.ll.pop_function ();

    return true;
}



LLVMGEN (llvm_gen_return)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() == 0);
    if (op.opname() == Strings::op_exit) {
        // If it's a real "exit", totally jump out of the shader instance.
        // The exit instance block will be created if it doesn't yet exist.
        rop.ll.op_branch (rop.llvm_exit_instance_block());
    } else {
        // If it's a "return", jump to the exit point of the function.
        rop.ll.op_branch (rop.ll.return_block());
    }
    llvm::BasicBlock* next_block = rop.ll.new_basic_block ("");
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
