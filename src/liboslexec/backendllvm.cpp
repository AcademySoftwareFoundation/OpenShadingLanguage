// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


// NEW - : Nagłówki LLVM potrzebne do zapisu bitkodu dla AMDGPU ---
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>
#include <map>
#include <string>
#include <optional>
#include <iostream>
// Nagłówki Targetu i Emisji Kodu dla LLVM 18
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/CodeGen.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>

#include "oslexec_pvt.h"
#include "backendllvm.h"



#include <fstream>

#include <map>
#include <string>
// Globalny cache testowy 
std::map<std::string, std::string> g_amdgpu_temp_cache;

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_BEGIN

namespace pvt {


BackendLLVM::BackendLLVM(ShadingSystemImpl& shadingsys, ShaderGroup& group,
                         ShadingContext* ctx)
    : OSOProcessorBase(shadingsys, group, ctx)
    , ll(ctx->llvm_thread_info(), llvm_debug(), shadingsys.m_vector_width)
    , m_stat_total_llvm_time(0)
    , m_stat_llvm_setup_time(0)
    , m_stat_llvm_irgen_time(0)
    , m_stat_llvm_opt_time(0)
    , m_stat_llvm_jit_time(0)
{
    m_use_optix      = shadingsys.use_optix();
    m_use_rs_bitcode = !shadingsys.m_rs_bitcode.empty();
    m_name_llvm_syms = shadingsys.m_llvm_output_bitcode;

    // Select the appropriate ustring representation
    ll.ustring_rep(LLVM_Util::UstringRep::hash);

    ll.dumpasm(shadingsys.m_llvm_dumpasm);
    ll.jit_fma(shadingsys.m_llvm_jit_fma);
    ll.jit_aggressive(shadingsys.m_llvm_jit_aggressive);
}



BackendLLVM::~BackendLLVM() {}



int
BackendLLVM::llvm_debug() const
{
    if (shadingsys().llvm_debug() == 0)
        return 0;
    if (!shadingsys().debug_groupname().empty()
        && shadingsys().debug_groupname() != group().name())
        return 0;
    if (inst() && !shadingsys().debug_layername().empty()
        && shadingsys().debug_layername() != inst()->layername())
        return 0;
    return shadingsys().llvm_debug();
}



void
BackendLLVM::set_inst(int layer)
{
    OSOProcessorBase::set_inst(layer);  // parent does the heavy lifting
    ll.debug(llvm_debug());
}



llvm::Type*
BackendLLVM::llvm_pass_type(const TypeSpec& typespec)
{
    if (typespec.is_closure_based())
        return (llvm::Type*)ll.type_void_ptr();
    TypeDesc t     = typespec.simpletype().elementtype();
    llvm::Type* lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = ll.type_float();
    else if (t == TypeDesc::INT)
        lt = ll.type_int();
    else if (t == TypeDesc::STRING)
        // When interpretting parameters, "s" is a real string
        // regardless of LLVM_Util::UStringRep
        // And "h" is used for hashes which maps to OSL::TypeUint64
        lt = (llvm::Type*)ll.type_real_ustring();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = (llvm::Type*)ll.type_void_ptr();  //llvm_type_triple_ptr();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = (llvm::Type*)ll.type_void_ptr();  //llvm_type_matrix_ptr();
    else if (t == TypeDesc::NONE)
        lt = ll.type_void();
    else if (t == TypeDesc::PTR)
        lt = (llvm::Type*)ll.type_void_ptr();
    else if (t == TypeDesc::LONGLONG)
        lt = ll.type_longlong();
    else if (t == OSL::TypeUInt64)
        lt = ll.type_int64();  //LLVM does not recognize signed bits
    else {
        OSL_ASSERT_MSG(0, "not handling %s type yet", typespec.c_str());
    }
    if (t.arraylen) {
        OSL_ASSERT(0 && "should never pass an array directly as a parameter");
    }
    return lt;
}



void
BackendLLVM::llvm_assign_zero(const Symbol& sym)
{
    // Just memset the whole thing to zero, let LLVM sort it out.
    // This even works for closures.
    int len;
    if (sym.typespec().is_closure_based())
        len = sizeof(void*) * sym.typespec().numelements();
    else
        len = sym.derivsize();
    // N.B. derivsize() includes derivs, if there are any
    size_t align = sym.typespec().is_closure_based()
                       ? sizeof(void*)
                       : sym.typespec().simpletype().basesize();
    ll.op_memset(llvm_void_ptr(sym), 0, len, (int)align);
}



void
BackendLLVM::llvm_zero_derivs(const Symbol& sym)
{
    if (sym.typespec().is_closure_based())
        return;  // Closures don't have derivs
    // Just memset the derivs to zero, let LLVM sort it out.
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_float_based()) {
        int len      = sym.size();
        size_t align = sym.typespec().simpletype().basesize();
        ll.op_memset(llvm_void_ptr(sym, 1), /* point to start of x deriv */
                     0, 2 * len /* size of both derivs */, (int)align);
    }
}



void
BackendLLVM::llvm_zero_derivs(const Symbol& sym, llvm::Value* count)
{
    if (sym.typespec().is_closure_based())
        return;  // Closures don't have derivs
    // Same thing as the above version but with just the first count derivs
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_float_based()) {
        size_t esize = sym.typespec().simpletype().elementsize();
        size_t align = sym.typespec().simpletype().basesize();
        count        = ll.op_mul(count, ll.constant((int)esize));
        ll.op_memset(llvm_void_ptr(sym, 1), 0, count, (int)align);  // X derivs
        ll.op_memset(llvm_void_ptr(sym, 2), 0, count, (int)align);  // Y derivs
    }
}

namespace {
// N.B. The order of names in this table MUST exactly match the
// ShaderGlobals struct in oslexec.h, as well as the llvm 'sg' type
// defined in llvm_type_sg().
static ustring fields[] = { ustring("P"),
                            ustring("_dPdz"),
                            ustring("I"),
                            ustring("N"),
                            ustring("Ng"),
                            ustring("u"),
                            ustring("v"),
                            ustring("dPdu"),
                            ustring("dPdv"),
                            ustring("time"),
                            ustring("dtime"),
                            ustring("dPdtime"),
                            ustring("Ps"),
                            ustring("renderstate"),
                            ustring("tracedata"),
                            ustring("objdata"),
                            ustring("shadingcontext"),
                            ustring("shadingStateUniform"),
                            ustring("thread_index"),
                            ustring("shade_index"),
                            ustring("renderer"),
                            ustring("object2common"),
                            ustring("shader2common"),
                            ustring("Ci"),
                            ustring("surfacearea"),
                            ustring("raytype"),
                            ustring("flipHandedness"),
                            ustring("backfacing") };
}  // namespace

int
BackendLLVM::ShaderGlobalNameToIndex(ustring name)
{
    for (int i = 0; i < int(sizeof(fields) / sizeof(fields[0])); ++i)
        if (name == fields[i])
            return i;
    return -1;
}



llvm::Value*
BackendLLVM::llvm_global_symbol_ptr(ustring name)
{
    // Special case for globals -- they live in the ShaderGlobals struct,
    // we use the name of the global to find the index of the field within
    // the ShaderGlobals struct.
    int sg_index = ShaderGlobalNameToIndex(name);
    OSL_ASSERT(sg_index >= 0);
    return ll.void_ptr(ll.GEP(llvm_type_sg(), sg_ptr(), 0, sg_index),
                       llnamefmt("glob_{}_voidptr", name));
}



llvm::Value*
BackendLLVM::getLLVMSymbolBase(const Symbol& sym)
{
    Symbol* dealiased = sym.dealias();

    if (sym.symtype() == SymTypeGlobal) {
        llvm::Value* result = llvm_global_symbol_ptr(sym.name());
        OSL_ASSERT(result);
        result = ll.ptr_to_cast(result,
                                llvm_type(sym.typespec().elementtype()));
        return result;
    }
    if (sym.symtype() == SymTypeParam && sym.interactive()) {
        // Special case for interactively-edited parameters -- they live in
        // the interactive data block for the group.
        // Generate the pointer to this symbol by offsetting into the
        // interactive data block.
        int offset = group().interactive_param_offset(layer(), sym.name());
        return ll.offset_ptr(m_llvm_interactive_params_ptr, offset,
                             llvm_ptr_type(sym.typespec().elementtype()));
    }

    if (sym.symtype() == SymTypeParam
        || (sym.symtype() == SymTypeOutputParam
            && !can_treat_param_as_local(sym))) {
        // Special case for most params -- they live in the group data
        int fieldnum = m_param_order_map[&sym];
        return groupdata_field_ptr(fieldnum,
                                   sym.typespec().elementtype().simpletype());
    }

    std::string mangled_name         = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);
    if (map_iter == named_values().end()) {
        shadingcontext()->errorfmt(
            "Couldn't find symbol '{}' (unmangled = '{}'). Did you forget to allocate it?",
            mangled_name, dealiased->unmangled());
        return 0;
    }
    return (llvm::Value*)map_iter->second;
}



llvm::Value*
BackendLLVM::llvm_alloca(const TypeSpec& type, bool derivs,
                         const std::string& name, int align)
{
    TypeDesc t = llvm_typedesc(type);
    int n      = derivs ? 3 : 1;
    m_llvm_local_mem += t.size() * n;
    return ll.op_alloca(t, n, name, align);
}


bool
BackendLLVM::can_treat_param_as_local(const Symbol& sym)
{
    if (!shadingsys().m_opt_groupdata)
        return false;

    // Some output parameters that are never needed before or
    // after layer execution can be relocated from GroupData
    // onto the stack.
    return sym.symtype() == SymTypeOutputParam && !sym.renderer_output()
           && !sym.typespec().is_closure_based() && !sym.connected();
}

llvm::Value*
BackendLLVM::getOrAllocateLLVMSymbol(const Symbol& sym)
{
    OSL_DASSERT(
        (sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp
         || sym.symtype() == SymTypeConst || can_treat_param_as_local(sym))
        && "getOrAllocateLLVMSymbol should only be for local, tmp, const");
    Symbol* dealiased                = sym.dealias();
    std::string mangled_name         = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);

    if (map_iter == named_values().end()) {
        llvm::Value* a = llvm_alloca(sym.typespec(), sym.has_derivs(),
                                     llnamefmt("{}_mem", mangled_name));
        named_values()[mangled_name] = a;
        return a;
    }
    return map_iter->second;
}


llvm::Value*
BackendLLVM::llvm_get_pointer(const Symbol& sym, int deriv,
                              llvm::Value* arrayindex)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Return NULL for request for pointer to derivs that don't exist
        return ll.ptr_cast(ll.void_ptr_null(),
                           ll.type_ptr(llvm_type(sym.typespec().elementtype())));
    }

    llvm::Value* result = NULL;
    if (sym.symtype() == SymTypeConst) {
        auto sym_name = sym.name().string();

        std::string unique_symname = global_unique_symname(sym);
        auto it                    = get_const_map().find(unique_symname);
        OSL_ASSERT(it != get_const_map().end());
        result = it->second;
        if (result) {
            TypeSpec elemtype = sym.typespec().elementtype();
            result            = llvm_ptr_cast(result, llvm_typedesc(elemtype),
                                              llnamefmt("cast_to_{}_", sym.typespec()));
        }
        return result;
    } else {
        // If the symbol is not a SymTypeConst, then start with the initial
        // pointer to the variable's memory location.
        result = getLLVMSymbolBase(sym);
    }
    if (!result)
        return NULL;  // Error

    // If it's an array or we're dealing with derivatives, step to the right
    // element.
    TypeDesc t = sym.typespec().simpletype();
    if (t.arraylen || has_derivs) {
        int d = deriv * std::max(1, t.arraylen);
        if (arrayindex && d)
            arrayindex = ll.op_add(arrayindex, ll.constant(d));
        else
            arrayindex = ll.constant(d);
        result = ll.GEP(llvm_type(t.elementtype()), result, arrayindex);
    }

    return result;
}



llvm::Value*
BackendLLVM::llvm_load_value(const Symbol& sym, int deriv,
                             llvm::Value* arrayindex, int component,
                             TypeDesc cast)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        return ll.constant(0.0f);
    }

    // arrayindex should be non-NULL if and only if sym is an array
    OSL_DASSERT(sym.typespec().is_array() == (arrayindex != NULL));

    if (sym.is_constant() && !sym.typespec().is_array() && !arrayindex) {
        // Shortcut for simple constants
        if (sym.typespec().is_float()) {
            if (cast == TypeInt)
                return ll.constant((int)sym.get_float());
            else
                return ll.constant(sym.get_float());
        }
        if (sym.typespec().is_int()) {
            if (cast == TypeFloat)
                return ll.constant((float)sym.get_int());
            else
                return ll.constant(sym.get_int());
        }
        if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
            return ll.constant(sym.get_float(component));
        }
        if (sym.typespec().is_string()) {
            return llvm_const_hash(sym.get_string());
        }
        OSL_ASSERT(0 && "unhandled constant type");
    }

    return llvm_load_value(llvm_get_pointer(sym), sym.typespec(), deriv,
                           arrayindex, component, cast,
                           llnamefmt("{}_", sym.name()));
}



llvm::Value*
BackendLLVM::llvm_load_value(llvm::Value* ptr, const TypeSpec& type, int deriv,
                             llvm::Value* arrayindex, int component,
                             TypeDesc cast, const std::string& llname)
{
    if (!ptr)
        return NULL;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t               = type.simpletype();
    llvm::Type* element_type = llvm_type(t.elementtype());
    if (t.arraylen || deriv) {
        int d = deriv * std::max(1, t.arraylen);
        if (arrayindex)
            arrayindex = ll.op_add(arrayindex, ll.constant(d));
        else
            arrayindex = ll.constant(d);
        ptr = ll.GEP(element_type, ptr, arrayindex);
    }

    // If it's multi-component (triple or matrix), step to the right field
    if (!type.is_closure_based() && t.aggregate > 1)
        ptr = ll.GEP(element_type, ptr, 0, component);

    // Now grab the value
    llvm::Type* component_type = llvm_type(t.scalartype());
    llvm::Value* result        = ll.op_load(component_type, ptr, llname);

    if (type.is_closure_based())
        return result;

    // Handle int<->float type casting
    if (type.is_float_based() && !type.is_array() && cast == TypeInt)
        result = ll.op_float_to_int(result);
    else if (type.is_int() && cast == TypeFloat)
        result = ll.op_int_to_float(result);

    return result;
}



llvm::Value*
BackendLLVM::llvm_load_constant_value(const Symbol& sym, int arrayindex,
                                      int component, TypeDesc cast)
{
    OSL_DASSERT(sym.is_constant()
                && "Called llvm_load_constant_value for a non-constant symbol");

    // set array indexing to zero for non-arrays
    if (!sym.typespec().is_array())
        arrayindex = 0;
    OSL_DASSERT(arrayindex >= 0
                && "Called llvm_load_constant_value with negative array index");

    int ncomps = (int)sym.typespec().aggregate();
    // Handle expanding single value to multiple.
    // Caller's responsiblity to keep component index in bounds otherwise
    if (ncomps == 1)
        component = 0;
    OSL_ASSERT(component < ncomps);
    int linear_index = ncomps * arrayindex + component;

    if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
        return ll.constant(sym.get_float(linear_index));
    }
    if (sym.typespec().is_float_based()) {
        if (cast == TypeInt)
            return ll.constant((int)sym.get_float(linear_index));
        else
            return ll.constant(sym.get_float(linear_index));
    }
    if (sym.typespec().is_int_based()) {
        if (cast == TypeFloat)
            return ll.constant((float)sym.get_int(linear_index));
        else
            return ll.constant(sym.get_int(linear_index));
    }
    if (sym.typespec().is_string_based()) {
        return llvm_const_hash(sym.get_string(linear_index));
    }

    OSL_ASSERT(0 && "unhandled constant type");
    return NULL;
}



llvm::Value*
BackendLLVM::llvm_load_component_value(const Symbol& sym, int deriv,
                                       llvm::Value* component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        OSL_DASSERT(sym.typespec().is_float_based()
                    && "can't ask for derivs of an int");
        return ll.constant(0.0f);
    }

    // Start with the initial pointer to the value's memory location
    llvm::Value* result = llvm_get_pointer(sym, deriv);
    if (!result)
        return NULL;  // Error

    OSL_DASSERT(sym.typespec().simpletype().aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*
    result = ll.ptr_cast(result, ll.type_float_ptr());
    result = ll.GEP(ll.type_float(), result, component);  // get the component

    // Now grab the value
    return ll.op_load(ll.type_float(), result);
}



llvm::Value*
BackendLLVM::llvm_load_arg(const Symbol& sym, bool derivs)
{
    OSL_DASSERT(sym.typespec().is_float_based());
    if (sym.typespec().is_int() || (sym.typespec().is_float() && !derivs)) {
        // Scalar case
        return llvm_load_value(sym);
    }

    if (derivs && !sym.has_derivs()) {
        // Manufacture-derivs case
        const TypeSpec& t = sym.typespec();
        // Copy the non-deriv values component by component
        llvm::Value* tmpptr = llvm_alloca(t, true);
        for (int c = 0; c < t.aggregate(); ++c) {
            llvm::Value* v = llvm_load_value(sym, 0, c);
            llvm_store_value(v, tmpptr, t, 0, NULL, c);
        }
        // Zero out the deriv values
        llvm::Value* zero = ll.constant(0.0f);
        for (int c = 0; c < t.aggregate(); ++c)
            llvm_store_value(zero, tmpptr, t, 1, NULL, c);
        for (int c = 0; c < t.aggregate(); ++c)
            llvm_store_value(zero, tmpptr, t, 2, NULL, c);
        return ll.void_ptr(tmpptr);
    }

    // Regular pointer case
    return llvm_void_ptr(sym);
}



bool
BackendLLVM::llvm_store_value(llvm::Value* new_val, const Symbol& sym,
                              int deriv, llvm::Value* arrayindex, int component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    return llvm_store_value(new_val, llvm_get_pointer(sym), sym.typespec(),
                            deriv, arrayindex, component);
}



bool
BackendLLVM::llvm_store_value(llvm::Value* new_val, llvm::Value* dst_ptr,
                              const TypeSpec& type, int deriv,
                              llvm::Value* arrayindex, int component)
{
    if (!dst_ptr)
        return false;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t               = type.simpletype();
    llvm::Type* element_type = llvm_type(t.elementtype());
    if (t.arraylen || deriv) {
        int d = deriv * std::max(1, t.arraylen);
        if (arrayindex)
            arrayindex = ll.op_add(arrayindex, ll.constant(d));
        else
            arrayindex = ll.constant(d);
        dst_ptr = ll.GEP(element_type, dst_ptr, arrayindex);
    }

    // If it's multi-component (triple or matrix), step to the right field
    if (!type.is_closure_based() && t.aggregate > 1)
        dst_ptr = ll.GEP(element_type, dst_ptr, 0, component);

    // Finally, store the value.
    // TODO: this breaks OptiX, pointer type comparison no longer works with opaque pointers.
    if (t == TypeString && dst_ptr->getType() == ll.type_int64_ptr()
        && new_val->getType() == ll.type_char_ptr()) {
        // Special case: we are still ickily storing strings sometimes as a
        // char* and sometimes as a uint64. Do a little sneaky conversion
        // here.
        new_val = ll.ptr_to_int64_cast(new_val);
    }
    ll.op_store(new_val, dst_ptr);
    return true;
}



bool
BackendLLVM::llvm_store_component_value(llvm::Value* new_val, const Symbol& sym,
                                        int deriv, llvm::Value* component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    // Let llvm_get_pointer do most of the heavy lifting to get us a
    // pointer to where our data lives.
    llvm::Value* result = llvm_get_pointer(sym, deriv);
    if (!result)
        return false;  // Error

    OSL_DASSERT(sym.typespec().simpletype().aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*
    result = ll.ptr_cast(result, ll.type_float_ptr());
    result = ll.GEP(ll.type_float(), result, component);  // get the component

    // Finally, store the value.
    ll.op_store(new_val, result);
    return true;
}



llvm::Value*
BackendLLVM::groupdata_field_ref(int fieldnum)
{
    return ll.GEP(llvm_type_groupdata(), groupdata_ptr(), 0, fieldnum,
                  llnamefmt("{}_ref", m_groupdata_field_names[fieldnum]));
}


llvm::Value*
BackendLLVM::groupdata_field_ptr(int fieldnum, TypeDesc type)
{
    llvm::Value* result = groupdata_field_ref(fieldnum);
    std::string llname = llnamefmt("{}_ptr", m_groupdata_field_names[fieldnum]);
    if (type != TypeDesc::UNKNOWN)
        result = ll.ptr_to_cast(result, llvm_type(type), llname);
    else
        result = ll.void_ptr(result, llname);
    return result;
}


llvm::Value*
BackendLLVM::layer_run_ref(int layer)
{
    int fieldnum = 0;  // field 0 is the layer_run array
    return ll.GEP(llvm_type_groupdata(), groupdata_ptr(), 0, fieldnum, layer,
                  llnamefmt("layer_runflags_ref"));
}



llvm::Value*
BackendLLVM::userdata_initialized_ref(int userdata_index)
{
    int fieldnum = 1;  // field 1 is the userdata_initialized array
    return ll.GEP(llvm_type_groupdata(), groupdata_ptr(), 0, fieldnum,
                  userdata_index, llnamefmt("userdata_init_flags_ref"));
}



llvm::Value*
BackendLLVM::llvm_call_function(const char* name, cspan<const Symbol*> args,
                                bool deriv_ptrs)
{
    // most invocations of this function will only need a handful of args
    // so avoid dynamic allocation where possible
    constexpr int SHORT_NUM_ARGS = 16;
    llvm::Value* short_valargs[SHORT_NUM_ARGS];
    std::vector<llvm::Value*> long_valargs;
    llvm::Value** valargs = short_valargs;
    if (args.size() > SHORT_NUM_ARGS) {
        long_valargs.resize(args.size());
        valargs = long_valargs.data();
    }
    for (int i = 0, nargs = args.size(); i < nargs; ++i) {
        const Symbol& s = *(args[i]);
        if (s.typespec().is_closure())
            valargs[i] = llvm_load_value(s);
        else if (s.typespec().simpletype().aggregate > 1
                 || (deriv_ptrs && s.has_derivs()))
            valargs[i] = llvm_void_ptr(s);
        else
            valargs[i] = llvm_load_value(s);
    }
    return ll.call_function(name, cspan<llvm::Value*>(valargs, args.size()));
}



llvm::Value*
BackendLLVM::llvm_test_nonzero(Symbol& val, bool test_derivs)
{
    const TypeSpec& ts(val.typespec());
    OSL_DASSERT(!ts.is_array() && !ts.is_closure() && !ts.is_string());
    TypeDesc t = ts.simpletype();

    // Handle int case -- guaranteed no derivs, no multi-component
    if (t == TypeInt)
        return ll.op_ne(llvm_load_value(val), ll.constant(0));

    // float-based
    int ncomps             = t.aggregate;
    int nderivs            = (test_derivs && val.has_derivs()) ? 3 : 1;
    llvm::Value* isnonzero = NULL;
    for (int d = 0; d < nderivs; ++d) {
        for (int c = 0; c < ncomps; ++c) {
            llvm::Value* v  = llvm_load_value(val, d, c);
            llvm::Value* nz = ll.op_ne(v, ll.constant(0.0f), true);
            if (isnonzero)  // multi-component/deriv: OR with running result
                isnonzero = ll.op_or(nz, isnonzero);
            else
                isnonzero = nz;
        }
    }
    return isnonzero;
}



bool
BackendLLVM::llvm_assign_impl(Symbol& Result, Symbol& Src, int arrayindex,
                              int srccomp, int dstcomp)
{
    OSL_DASSERT(!Result.typespec().is_structure());
    OSL_DASSERT(!Src.typespec().is_structure());

    const TypeSpec& result_t(Result.typespec());
    const TypeSpec& src_t(Src.typespec());

    llvm::Value* arrind = arrayindex >= 0 ? ll.constant(arrayindex) : NULL;

    if (Result.typespec().is_closure() || Src.typespec().is_closure()) {
        if (Src.typespec().is_closure()) {
            llvm::Value* srcval = llvm_load_value(Src, 0, arrind, 0);
            llvm_store_value(srcval, Result, 0, arrind, 0);
        } else {
            llvm::Value* null = ll.constant_ptr(NULL, ll.type_void_ptr());
            llvm_store_value(null, Result, 0, arrind, 0);
        }
        return true;
    }

    if (Result.typespec().is_matrix() && Src.typespec().is_int_or_float()) {
        // Handle m=f, m=i separately
        llvm::Value* src = llvm_load_value(Src, 0, arrind, 0,
                                           TypeDesc::FLOAT /*cast*/);
        // m=f sets the diagonal components to f, the others to zero
        llvm::Value* zero = ll.constant(0.0f);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                llvm_store_value(i == j ? src : zero, Result, 0, arrind,
                                 i * 4 + j);
        llvm_zero_derivs(Result);  // matrices don't have derivs currently
        return true;
    }

    // Copying of entire arrays.  But only for non-const source symbols
    // as we don't want to generate memcpy to host memory or to ustrings
    // when we are really using ustringhash for llvm gen.
    // It's ok if the array lengths don't match,
    // it will only copy up to the length of the smaller one.  The compiler
    // will ensure they are the same size, except for certain cases where
    // the size difference is intended (by the optimizer).
    if (result_t.is_array() && !Src.is_constant() && src_t.is_array()
        && arrayindex == -1) {
        OSL_DASSERT(assignable(result_t.elementtype(), src_t.elementtype()));
        llvm::Value* resultptr = llvm_get_pointer(Result);
        llvm::Value* srcptr    = llvm_get_pointer(Src);
        int len                = std::min(Result.size(), Src.size());
        int align              = result_t.is_closure_based()
                                     ? (int)sizeof(void*)
                                     : (int)result_t.simpletype().basesize();
        if (Result.has_derivs() && Src.has_derivs()) {
            ll.op_memcpy(resultptr, srcptr, 3 * len, align);
        } else {
            ll.op_memcpy(resultptr, srcptr, len, align);
            if (Result.has_derivs())
                llvm_zero_derivs(Result);
        }
        return true;
    }

    // The following code handles f=f, f=i, v=v, v=f, v=i, m=m, s=s.
    // Remember that llvm_load_value will automatically convert scalar->triple.
    TypeDesc rt              = Result.typespec().simpletype();
    TypeDesc basetype        = TypeDesc::BASETYPE(rt.basetype);
    const int num_components = rt.aggregate;
    const bool singlechan    = (srccomp != -1) || (dstcomp != -1);
    if (!singlechan) {
        if (rt.is_array() && arrayindex == -1) {
            // Initialize entire array
            const int num_elements = std::min(rt.numelements(),
                                              src_t.simpletype().numelements());
            for (int a = 0; a < num_elements; ++a) {
                llvm::Value* const_arrind = ll.constant(a);
                for (int i = 0; i < num_components; ++i) {
                    llvm::Value* src_val
                        = Src.is_constant()
                              ? llvm_load_constant_value(Src, a, i, basetype)
                              : llvm_load_value(
                                  Src, 0,
                                  Src.typespec().is_array() ? const_arrind
                                                            : nullptr,
                                  (Src.typespec().aggregate() == 1) ? 0 : i,
                                  basetype);
                    if (!src_val)
                        return false;
                    llvm_store_value(src_val, Result, 0, const_arrind, i);
                }
            }
        } else {
            for (int i = 0; i < num_components; ++i) {
                llvm::Value* src_val
                    = Src.is_constant()
                          ? llvm_load_constant_value(Src, arrayindex, i,
                                                     basetype)
                          : llvm_load_value(Src, 0, arrind, i, basetype);
                if (!src_val)
                    return false;
                llvm_store_value(src_val, Result, 0, arrind, i);
            }
        }
    } else {
        // connect individual component of an aggregate type
        // set srccomp to 0 for case when src is actually a float
        if (srccomp == -1)
            srccomp = 0;
        llvm::Value* src_val
            = Src.is_constant()
                  ? llvm_load_constant_value(Src, arrayindex, srccomp, basetype)
                  : llvm_load_value(Src, 0, arrind, srccomp, basetype);
        if (!src_val)
            return false;
        // write source float into all components when dstcomp == -1, otherwise
        // the single element requested.
        if (dstcomp == -1) {
            for (int i = 0; i < num_components; ++i)
                llvm_store_value(src_val, Result, 0, arrind, i);
        } else
            llvm_store_value(src_val, Result, 0, arrind, dstcomp);
    }

    // Handle derivatives
    if (Result.has_derivs()) {
        if (Src.has_derivs()) {
            // src and result both have derivs -- copy them
            if (!singlechan) {
                for (int d = 1; d <= 2; ++d) {
                    for (int i = 0; i < num_components; ++i) {
                        llvm::Value* val = llvm_load_value(Src, d, arrind, i);
                        llvm_store_value(val, Result, d, arrind, i);
                    }
                }
            } else {
                for (int d = 1; d <= 2; ++d) {
                    llvm::Value* val = llvm_load_value(Src, d, arrind, srccomp);
                    if (dstcomp == -1) {
                        for (int i = 0; i < num_components; ++i)
                            llvm_store_value(val, Result, d, arrind, i);
                    } else
                        llvm_store_value(val, Result, d, arrind, dstcomp);
                }
            }
        } else {
            // Result wants derivs but src didn't have them -- zero them
            if (dstcomp != -1) {
                // memset the single deriv component's to zero
                if (Result.has_derivs()
                    && Result.typespec().elementtype().is_float_based()) {
                    // dx
                    ll.op_memset(ll.GEP(ll.type_void_ptr(),
                                        llvm_void_ptr(Result, 1), dstcomp),
                                 0, 1, rt.basesize());
                    // dy
                    ll.op_memset(ll.GEP(ll.type_void_ptr(),
                                        llvm_void_ptr(Result, 2), dstcomp),
                                 0, 1, rt.basesize());
                }
            } else
                llvm_zero_derivs(Result);
        }
    }
    return true;
}



int
BackendLLVM::find_userdata_index(const Symbol& sym)
{
    int userdata_index = -1;
    for (int i = 0, e = (int)group().m_userdata_names.size(); i < e; ++i) {
        if (sym.name() == group().m_userdata_names[i]
            && equivalent(sym.typespec().simpletype(),
                          group().m_userdata_types[i])) {
            userdata_index = i;
            break;
        }
    }
    return userdata_index;
}

std::vector<uint8_t> BackendLLVM::get_llvm_bitcode(llvm::Module* custom_mod) {
    shadingsys().info("=========================================");
    shadingsys().info("[AMD] ROZPOCZYNAM GENEROWANIE KODU DLA GPU");
    shadingsys().info("=========================================");
    
    // 1. WYBÓR WŁAŚCIWEGO MODUŁU:
    llvm::Module* mod = custom_mod ? custom_mod : ll.module();

    // 2. INICJALIZACJA BACKENDU AMDGPU W LLVM
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();

    // 3. DETEKCJA ARCHITEKTURY SPRZĘTOWEJ RDNA
    std::string arch_str = shadingsys().amdgpu_architecture().string();
    if (arch_str.empty()) {
        arch_str = "gfx1100"; // Fallback dla RDNA3
    }

    // 4. WYSZUKANIE TARGETU ORAZ KONFIGURACJA TARGET MACHINE
    std::string llvm_error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget("amdgcn-amd-amdhsa", llvm_error);
    if (!target) {
        shadingsys().error(OIIO::Strutil::format("LLVM Error: Nie znaleziono targetu AMDGPU: %s", llvm_error.c_str()));
        return std::vector<uint8_t>();
    }

    llvm::TargetOptions opt;
    llvm::TargetMachine* target_machine = target->createTargetMachine(
        "amdgcn-amd-amdhsa", arch_str, "", opt, llvm::Reloc::PIC_,
        llvm::CodeModel::Small, llvm::CodeGenOptLevel::None);

    if (!target_machine) {
        shadingsys().error("LLVM Error: Nie udalo sie utworzyc TargetMachine dla AMDGPU");
        return std::vector<uint8_t>();
    }

    // Dostosowanie układu pamięci modułu pod architekturę AMD
    mod->setDataLayout(target_machine->createDataLayout());
    mod->setTargetTriple("amdgcn-amd-amdhsa");

    // Wymuszenie wygenerowania HSA Kernel Descriptors przez LLVM
    mod->addModuleFlag(llvm::Module::Error, "amdgpu_code_object_version", 500);

// ==================== DEBUG AMD ====================
std::cout << "\n[AMD DEBUG] === ROZPOCZĘCIE ZRZUTU FUNKCJI W MODULE ===\n";
std::cout << "[AMD DEBUG] Nazwa modułu: " << (mod->getModuleIdentifier()) << "\n";
int func_counter = 0;
for (llvm::Function& F : *mod) {
    func_counter++;
    std::cout << "  [" << func_counter << "] Nazwa: " << F.getName().str()
              << " | Deklaracja: " << (F.isDeclaration() ? "TAK" : "NIE");
    if (!F.isDeclaration()) {
        std::cout << " | Instrukcji: " << F.getInstructionCount();
    }
    std::cout << "\n";
}
std::cout << "[AMD DEBUG] === KONIEC ZRZUTU (Razem funkcji: " << func_counter << ") ===\n\n";
// ==============================================================
// 5. LOKALIZACJA FUNKCJI OSL (Inicjalizacja + Warstwa)
    shadingsys().info("[LLVM AMDGPU] Budowanie Megakernela GPU: szukanie init i layer...");
    
    llvm::Function* osl_init_func = nullptr;
    llvm::Function* osl_layer_func = nullptr;

    for (llvm::Function &F : *mod) {
        if (!F.isDeclaration()) {
            std::string fname = F.getName().str();
            if (fname.find("osl_init_group") != std::string::npos) {
                osl_init_func = &F;
            } else if (fname.find("osl_layer_group") != std::string::npos) {
                osl_layer_func = &F;
            }
        }
    }

    llvm::Function* ref_func = osl_init_func ? osl_init_func : osl_layer_func;
    if (!ref_func) {
        shadingsys().error("[LLVM AMDGPU] BLAD! Nie znaleziono zadnej funkcji OSL do owrapowania!");
        return std::vector<uint8_t>();
    }

    shadingsys().info("[LLVM AMDGPU] Znaleziono funkcje OSL. Tworzenie wrappera...");

    // 6. DYNAMICZNA BUDOWA WRAPPERA (LAUNCHER KERNEL) O NAZWIE "osl_kernel"
    llvm::LLVMContext &ctx = mod->getContext();
    llvm::FunctionType *orig_type = ref_func->getFunctionType();
    std::vector<llvm::Type*> wrapper_param_types;

    for (llvm::Type *param_type : orig_type->params()) {
        if (param_type->isPointerTy()) {
            wrapper_param_types.push_back(llvm::PointerType::get(ctx, 1)); 
        } else {
            wrapper_param_types.push_back(param_type);
        }
    }
    wrapper_param_types.push_back(llvm::Type::getInt32Ty(ctx)); // + int width

    llvm::FunctionType *wrapper_type = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapper_param_types, false);
    llvm::Function *wrapper_func = llvm::Function::Create(wrapper_type, llvm::GlobalValue::ExternalLinkage, "osl_kernel", mod);
    wrapper_func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    wrapper_func->addFnAttr("amdgpu-flat-work-group-size", "1,1024");

    llvm::BasicBlock *entry_block = llvm::BasicBlock::Create(ctx, "entry", wrapper_func);
    llvm::IRBuilder<> builder(entry_block);

    // 7. WSTRZYKIWANIE KOORDYNATÓW WĄTKÓW
    llvm::FunctionType *i32_fn_type = llvm::FunctionType::get(builder.getInt32Ty(), false);
    llvm::FunctionCallee wg_id_x_c = mod->getOrInsertFunction("llvm.amdgcn.workgroup.id.x", i32_fn_type);
    llvm::FunctionCallee wg_id_y_c = mod->getOrInsertFunction("llvm.amdgcn.workgroup.id.y", i32_fn_type);
    llvm::FunctionCallee wi_id_x_c = mod->getOrInsertFunction("llvm.amdgcn.workitem.id.x", i32_fn_type);
    llvm::FunctionCallee wi_id_y_c = mod->getOrInsertFunction("llvm.amdgcn.workitem.id.y", i32_fn_type);

    llvm::Value *wg_id_x = builder.CreateCall(wg_id_x_c);
    llvm::Value *wg_id_y = builder.CreateCall(wg_id_y_c);
    llvm::Value *wi_id_x = builder.CreateCall(wi_id_x_c);
    llvm::Value *wi_id_y = builder.CreateCall(wi_id_y_c);

    llvm::Value *block_dim = builder.getInt32(16);
    llvm::Value *global_x = builder.CreateAdd(builder.CreateMul(wg_id_x, block_dim), wi_id_x);
    llvm::Value *global_y = builder.CreateAdd(builder.CreateMul(wg_id_y, block_dim), wi_id_y);

    llvm::Argument *width_arg = wrapper_func->getArg(ref_func->arg_size()); 
    llvm::Value *pixel_index = builder.CreateAdd(builder.CreateMul(global_y, width_arg), global_x);

    // 8. OBLICZANIE WSKAŹNIKÓW I WYWOŁANIE FUNKCJI
    std::vector<llvm::Value*> call_args;
    auto orig_arg_it = ref_func->arg_begin();
    size_t arg_idx = 0;

    // Pobieramy rozmiar struktury GroupData (zabezpieczone min. 16 bajtów)
    int gd_size = std::max((int)group().llvm_groupdata_size(), 16);

    for (llvm::Argument &wrap_arg : wrapper_func->args()) {
        if (arg_idx >= ref_func->arg_size()) break;

        llvm::Value *final_val = &wrap_arg;

        if (wrap_arg.getType()->isPointerTy()) {
            if (arg_idx == 1) { 
                llvm::Value *local_gd = builder.CreateAlloca(builder.getInt8Ty(), builder.getInt32(gd_size), "local_group_data");
                final_val = builder.CreateAddrSpaceCast(local_gd, orig_arg_it->getType(), "cast_to_flat");
            } else {
                llvm::Value *offset = nullptr;
                if (arg_idx == 0) { 
                    offset = builder.CreateMul(pixel_index, builder.getInt32(256));
                } else if (arg_idx == 3) { 
                    offset = nullptr; 
                }

                if (offset) {
                    final_val = builder.CreateGEP(builder.getInt8Ty(), &wrap_arg, offset);
                }
                final_val = builder.CreateAddrSpaceCast(final_val, orig_arg_it->getType(), "cast_to_flat");
            }
        } 
       
        else if (arg_idx == 4) {
            final_val = pixel_index;
        }
        
        call_args.push_back(final_val);
        orig_arg_it++;
        arg_idx++;
    }

    // SEKWENCYJNE WYWOŁANIE FUNKCJI OSL
    if (osl_init_func) {
        builder.CreateCall(orig_type, osl_init_func, call_args);
    }
    if (osl_layer_func) {
        builder.CreateCall(orig_type, osl_layer_func, call_args);
    }

    builder.CreateRetVoid();
    
    // 8.5 WYMUSZENIE WIDOCZNOŚCI I ATRYBUTÓW KERNELA (Dla LLVM 18+)
    // 1. Zmieniamy całą tożsamość modułu na AMDGPU
    mod->setTargetTriple(target_machine->getTargetTriple().str());
    mod->setDataLayout(target_machine->createDataLayout());

    // 2. Oznaczamy nasz wrapper jako KERNEL sprzętowy
    wrapper_func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    wrapper_func->setLinkage(llvm::GlobalValue::ExternalLinkage);
    wrapper_func->setVisibility(llvm::GlobalValue::DefaultVisibility);
    wrapper_func->addFnAttr("amdgpu-kernel"); 

    shadingsys().info("[LLVM AMDGPU] ---> Tożsamość modułu zmieniona na AMDGPU. Wrapper ustawiony jako KERNEL.");

    // DIAGNOZA MODUŁU LLVM - SPRAWDZAMY, CZY FUNKCJE MAJĄ CIAŁA!

    llvm::errs() << "\n[!!! DIAGNOZA LLVM !!!]\n";
    llvm::errs() << "Liczba wszystkich funkcji w module: " << mod->getFunctionList().size() << "\n";
    
    for (llvm::Function &F : *mod) {
        // Wypisujemy tylko interesujące nas funkcje (nasz kernel i oryginalny kod shadera)
        if (F.getName() == "osl_kernel" || F.getName().contains("group_") || F.getName().contains("shader")) {
            llvm::errs() << " -> Funkcja: " << F.getName() 
                         << " | Czy jest pusta (tylko deklaracja)? " << (F.empty() ? "TAK (Brak kodu!)" : "NIE (Ma kod!)") << "\n";
        }
    }
    llvm::errs() << "[!!! KONIEC DIAGNOZY !!!]\n\n";

    // 8.6 ZRZUT WYGENEROWANEGO KODU IR DO PLIKU 

    std::error_code EC;
    llvm::raw_fd_ostream ir_file("/tmp/osl_ir_dump.ll", EC);
    if (!EC) {
        mod->print(ir_file, nullptr);
        ir_file.close();
        shadingsys().info("[LLVM AMDGPU] Zapisano zrzut kodu LLVM IR do /tmp/osl_ir_dump.ll");
    } else {
        shadingsys().error("[LLVM AMDGPU] Nie udalo sie zapisac pliku IR dump.");
    }

    // Ukrywamy oryginalne funkcje, żeby LLVM skupił się na wyeksportowaniu tylko wrappera
    if (osl_init_func) osl_init_func->setLinkage(llvm::GlobalValue::InternalLinkage);
    if (osl_layer_func) osl_layer_func->setLinkage(llvm::GlobalValue::InternalLinkage);
    // 8.7 USUWANIE ATRYBUTÓW CPU I WERYFIKACJA 
    for (llvm::Function &F : *mod) {
        F.removeFnAttr("target-cpu");
        F.removeFnAttr("target-features");
        F.removeFnAttr("tune-cpu");
        
        F.removeFnAttr(llvm::Attribute::OptimizeNone);
        F.removeFnAttr(llvm::Attribute::NoInline);
    }

    shadingsys().info("[LLVM AMDGPU] Atrybuty CPU zostały usunięte. Rozpoczynam weryfikację kodu IR...");

    // 9. EMISJA KODU
    llvm::SmallVector<char, 4096> elf_buffer;
    llvm::raw_svector_ostream dest(elf_buffer);

    llvm::legacy::PassManager code_gen_pm;
    if (target_machine->addPassesToEmitFile(code_gen_pm, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        shadingsys().error("LLVM Error: Backend kompilatora LLVM nie wspiera bezpośredniej emisji ELF dla AMDGPU");
        return std::vector<uint8_t>();
    }

    std::cout << "[DEBUG-LLVM] 3. Uruchamiam code_gen_pm.run(*mod) -> To moze chwile potrwac..."<<std::endl;

    // Generowanie kodu maszynowego
    code_gen_pm.run(*mod);

    std::cout << "[DEBUG-LLVM] 4. Sukces emisji kodu do bufora. Zapisuje do pliku /tmp/osl_temp_shader.o..."<<std::endl;

    // 10. CACHOWANIE I ZAPIS
    std::string cache_value(elf_buffer.begin(), elf_buffer.end());
    std::ofstream out_file("/tmp/osl_temp_shader.o", std::ios::binary);
    if (out_file.is_open()) {
        out_file.write(cache_value.data(), cache_value.size());
        out_file.close();
        std::cout << "[LLVM AMDGPU] Zapisano gotowy obiekt ELF do /tmp/osl_temp_shader.o\n";
    } else {
        std::cerr << "[LLVM AMDGPU] BLAD! Nie moglem zapisac do /tmp/osl_temp_shader.o\n";
    }

    // --- REJESTRACJA W PAMIĘCI ---
    extern std::map<const void*, std::vector<uint8_t>> g_amdgpu_elf_registry;
    extern std::mutex g_amdgpu_registry_mutex;

    std::vector<uint8_t> elf_blob(elf_buffer.begin(), elf_buffer.end());
    {
        std::lock_guard<std::mutex> lock(g_amdgpu_registry_mutex);
        g_amdgpu_elf_registry[&group()] = elf_blob; 
    }
    return elf_blob;
}}
// namespace pvt
OSL_NAMESPACE_END
