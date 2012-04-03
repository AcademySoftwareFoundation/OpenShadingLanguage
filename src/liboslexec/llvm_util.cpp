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

#include "llvm_headers.h"

#include "oslexec_pvt.h"
#include "runtimeoptimize.h"

using namespace OSL;
using namespace OSL::pvt;

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


llvm::Type *
RuntimeOptimizer::llvm_type_union(const std::vector<llvm::Type *> &types)
{
    llvm::TargetData target(llvm_module());
    size_t max_size = 0;
    size_t max_align = 1;
    for (size_t i = 0; i < types.size(); ++i) {
        size_t size = target.getTypeStoreSize(types[i]);
        size_t align = target.getABITypeAlignment(types[i]);
        max_size  = size  > max_size  ? size  : max_size;
        max_align = align > max_align ? align : max_align;
    }
    size_t padding = (max_size % max_align) ? max_align - (max_size % max_align) : 0;
    size_t union_size = max_size + padding;

    llvm::Type * base_type = NULL;
    // to ensure the alignment when included in a struct use
    // an appropiate type for the array
    if (max_align == sizeof(void*))
        base_type = llvm_type_void_ptr();
    else if (max_align == 4)
        base_type = (llvm::Type *) llvm::Type::getInt32Ty (llvm_context());
    else if (max_align == 2)
        base_type = (llvm::Type *) llvm::Type::getInt16Ty (llvm_context());
    else
        base_type = (llvm::Type *) llvm::Type::getInt8Ty (llvm_context());

    size_t array_len = union_size / target.getTypeStoreSize(base_type);
    return (llvm::Type *) llvm::ArrayType::get (base_type, array_len);
}



llvm::Type *
RuntimeOptimizer::llvm_type_struct (const std::vector<llvm::Type *> &types,
                                    const std::string &name)
{
#if OSL_LLVM_VERSION <= 29
    return (llvm::Type *) llvm::StructType::get(llvm_context(),
                            *(std::vector<const llvm::Type*>*)&types);
#else
    return llvm::StructType::create(llvm_context(), types, name);
#endif
}



llvm::Value *
RuntimeOptimizer::llvm_constant (float f)
{
    return llvm::ConstantFP::get (llvm_context(), llvm::APFloat(f));
}



llvm::Value *
RuntimeOptimizer::llvm_constant (int i)
{
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(32,i));
}



llvm::Value *
RuntimeOptimizer::llvm_constant (size_t i)
{
    int bits = sizeof(size_t)*8;
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(bits,i));
}



llvm::Value *
RuntimeOptimizer::llvm_constant_bool (bool i)
{
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(1,i));
}



llvm::Value *
RuntimeOptimizer::llvm_constant (ustring s)
{
    // Create a const size_t with the ustring contents
    size_t bits = sizeof(size_t)*8;
    llvm::Value *str = llvm::ConstantInt::get (llvm_context(),
                               llvm::APInt(bits,size_t(s.c_str()), true));
    // Then cast the int to a char*.
    return builder().CreateIntToPtr (str, llvm_type_string(), "ustring constant");
}



llvm::Value *
RuntimeOptimizer::llvm_constant_ptr (void *p)
{
    // Create a const size_t with the address
    size_t bits = sizeof(size_t)*8;
    llvm::Value *str = llvm::ConstantInt::get (llvm_context(),
                               llvm::APInt(bits,size_t(p), true));
    // Then cast the size_t to a char*.
    return builder().CreateIntToPtr (str, llvm_type_void_ptr());
}



llvm::Value *
RuntimeOptimizer::llvm_constant (const TypeDesc &type)
{
    long long *i = (long long *)&type;
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(64,*i));
}



llvm::Type *
RuntimeOptimizer::llvm_type (const TypeSpec &typespec)
{
    if (typespec.is_closure_based())
        return llvm_type_void_ptr();
    TypeDesc t = typespec.simpletype().elementtype();
    llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = llvm_type_float();
    else if (t == TypeDesc::INT)
        lt = llvm_type_int();
    else if (t == TypeDesc::STRING)
        lt = llvm_type_string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = llvm_type_triple();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = llvm_type_matrix();
    else if (t == TypeDesc::NONE)
        lt = llvm_type_void();
    else if (t == TypeDesc::PTR)
        lt = llvm_type_void_ptr();
    else {
        std::cerr << "Bad llvm_type(" << typespec.c_str() << ")\n";
        ASSERT (0 && "not handling this type yet");
    }
    if (typespec.is_array())
        lt = llvm::ArrayType::get (lt, typespec.simpletype().numelements());
    return lt;
}



llvm::Type *
RuntimeOptimizer::llvm_pass_type (const TypeSpec &typespec)
{
    if (typespec.is_closure_based())
        return llvm_type_void_ptr();
    TypeDesc t = typespec.simpletype().elementtype();
    llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = llvm_type_float();
    else if (t == TypeDesc::INT)
        lt = llvm_type_int();
    else if (t == TypeDesc::STRING)
        lt = llvm_type_string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = llvm_type_void_ptr(); //llvm_type_triple_ptr();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = llvm_type_void_ptr(); //llvm_type_matrix_ptr();
    else if (t == TypeDesc::NONE)
        lt = llvm_type_void();
    else if (t == TypeDesc::PTR)
        lt = llvm_type_void_ptr();
    else if (t == TypeDesc::LONGLONG)
        lt = llvm_type_longlong();
    else {
        std::cerr << "Bad llvm_pass_type(" << typespec.c_str() << ")\n";
        ASSERT (0 && "not handling this type yet");
    }
    if (t.arraylen) {
        ASSERT (0 && "should never pass an array directly as a parameter");
    }
    return lt;
}



void
RuntimeOptimizer::llvm_assign_zero (const Symbol &sym)
{
    // Just memset the whole thing to zero, let LLVM sort it out.
    // This even works for closures.
    int len = sym.typespec().is_closure_based() ? sizeof(void *) : sym.derivsize();
    // N.B. derivsize() includes derivs, if there are any
    size_t align = sym.typespec().is_closure_based() ? sizeof(void*) :
                         sym.typespec().simpletype().basesize();
    llvm_memset (llvm_void_ptr(sym), 0, len, (int)align);
}



void
RuntimeOptimizer::llvm_zero_derivs (const Symbol &sym)
{
    if (sym.typespec().is_closure_based())
        return; // Closures don't have derivs
    // Just memset the derivs to zero, let LLVM sort it out.
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_floatbased()) {
        int len = sym.size();
        size_t align = sym.typespec().simpletype().basesize();
        llvm_memset (llvm_void_ptr(sym,1), /* point to start of x deriv */
                     0, 2*len /* size of both derivs */, (int)align);
    }
}



void
RuntimeOptimizer::llvm_zero_derivs (const Symbol &sym, llvm::Value *count)
{
    if (sym.typespec().is_closure_based())
        return; // Closures don't have derivs
    // Same thing as the above version but with just the first count derivs
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_floatbased()) {
        size_t esize = sym.typespec().simpletype().elementsize();
        size_t align = sym.typespec().simpletype().basesize();
        count = builder().CreateMul(count, llvm_constant((int)esize));
        llvm_memset (llvm_void_ptr(sym,1), 0, count, (int)align); // X derivs
        llvm_memset (llvm_void_ptr(sym,2), 0, count, (int)align); // Y derivs
    }
}



int
RuntimeOptimizer::ShaderGlobalNameToIndex (ustring name)
{
    // N.B. The order of names in this table MUST exactly match the
    // ShaderGlobals struct in oslexec.h, as well as the llvm 'sg' type
    // defined in llvm_type_sg().
    static ustring fields[] = {
        Strings::P, ustring("_dPdz"), Strings::I, Strings::N, Strings::Ng,
        Strings::u, Strings::v, Strings::dPdu, Strings::dPdv,
        Strings::time, Strings::dtime, Strings::dPdtime, Strings::Ps,
        ustring("renderstate"), ustring("tracedata"), ustring("objdata"),
        ustring("shadingcontext"),
        ustring("object2common"), ustring("shader2common"),
        Strings::Ci,
        ustring("surfacearea"), ustring("raytype"),
        ustring("flipHandedness"), ustring("backfacing")
    };

    for (int i = 0;  i < int(sizeof(fields)/sizeof(fields[0]));  ++i)
        if (name == fields[i])
            return i;
    return -1;
}



llvm::Value *
RuntimeOptimizer::getLLVMSymbolBase (const Symbol &sym)
{
    Symbol* dealiased = sym.dealias();

    if (sym.symtype() == SymTypeGlobal) {
        // Special case for globals -- they live in the shader globals struct
        int sg_index = ShaderGlobalNameToIndex (sym.name());
        ASSERT (sg_index >= 0);
        llvm::Value *result = builder().CreateConstGEP2_32 (sg_ptr(), 0, sg_index);
        // No derivs?  We're one indirection too few?
        result = builder().CreatePointerCast (result, llvm::PointerType::get(llvm_type(sym.typespec().elementtype()), 0));
        return result;
    }

    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        // Special case for params -- they live in the group data
        int fieldnum = m_param_order_map[&sym];
        llvm::Value *result = builder().CreateConstGEP2_32 (groupdata_ptr(), 0,
                                                            fieldnum);
        // No derivs?  We're one indirection too few?
        result = builder().CreatePointerCast (result, llvm::PointerType::get(llvm_type(sym.typespec().elementtype()), 0));
        return result;
    }

    std::string mangled_name = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find (mangled_name);
    if (map_iter == named_values().end()) {
        shadingsys().error ("Couldn't find symbol '%s' (unmangled = '%s'). Did you forget to allocate it?",
                            mangled_name.c_str(), dealiased->name().c_str());
        return 0;
    }
    return map_iter->second;
}



llvm::AllocaInst *
RuntimeOptimizer::llvm_alloca (const TypeSpec &type, bool derivs,
                               const std::string &name)
{
    TypeSpec elemtype = type.elementtype();
    llvm::Type *alloctype = llvm_type (elemtype);
    int arraylen = std::max (1, type.arraylength());
    int n = arraylen * (derivs ? 3 : 1);
    size_t size = type.is_closure() ? sizeof(void *)*arraylen : type.simpletype().size();
    m_llvm_local_mem += size * (derivs ? 3 : 1);
    llvm::ConstantInt* numalloc = (llvm::ConstantInt*)llvm_constant(n);
    return builder().CreateAlloca(alloctype, numalloc, name);
}



llvm::Value *
RuntimeOptimizer::getOrAllocateLLVMSymbol (const Symbol& sym)
{
    DASSERT ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp ||
              sym.symtype() == SymTypeConst)
             && "getOrAllocateLLVMSymbol should only be for local, tmp, const");
    Symbol* dealiased = sym.dealias();
    std::string mangled_name = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);

    if (map_iter == named_values().end()) {
        llvm::AllocaInst* allocation =
            llvm_alloca (sym.typespec(), sym.has_derivs(), mangled_name);
        named_values()[mangled_name] = allocation;
        return allocation;
    }
    return map_iter->second;
}



llvm::Value *
RuntimeOptimizer::llvm_get_pointer (const Symbol& sym, int deriv,
                                    llvm::Value *arrayindex)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Return NULL for request for pointer to derivs that don't exist
        return llvm_ptr_cast (llvm_void_ptr_null(),
                              llvm::PointerType::get (llvm_type(sym.typespec().elementtype()), 0));
    }

    llvm::Value *result = NULL;
    if (sym.symtype() == SymTypeConst) {
        // For constants, start with *OUR* pointer to the constant values.
        result = llvm_ptr_cast (llvm_constant_ptr (sym.data()),
                                llvm::PointerType::get (llvm_type(sym.typespec().elementtype()), 0));

    } else {
        // Start with the initial pointer to the variable's memory location
        result = getLLVMSymbolBase (sym);
    }
    if (!result)
        return NULL;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t = sym.typespec().simpletype();
    if (t.arraylen || has_derivs) {
        int d = deriv * std::max(1,t.arraylen);
        if (arrayindex)
            arrayindex = builder().CreateAdd (arrayindex, llvm_constant(d));
        else
            arrayindex = llvm_constant(d);
        result = builder().CreateGEP (result, arrayindex);
    }

    return result;
}



llvm::Value *
RuntimeOptimizer::llvm_load_value (const Symbol& sym, int deriv,
                                   llvm::Value *arrayindex, int component,
                                   TypeDesc cast)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        return llvm_constant (0.0f);
    }

    // arrayindex should be non-NULL if and only if sym is an array
    ASSERT (sym.typespec().is_array() == (arrayindex != NULL));

    if (sym.is_constant() && !sym.typespec().is_array() && !arrayindex) {
        // Shortcut for simple constants
        if (sym.typespec().is_float()) {
            if (cast == TypeDesc::TypeInt)
                return llvm_constant ((int)*(float *)sym.data());
            else
                return llvm_constant (*(float *)sym.data());
        }
        if (sym.typespec().is_int()) {
            if (cast == TypeDesc::TypeFloat)
                return llvm_constant ((float)*(int *)sym.data());
            else
                return llvm_constant (*(int *)sym.data());
        }
        if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
            return llvm_constant (((float *)sym.data())[component]);
        }
        if (sym.typespec().is_string()) {
            return llvm_constant (*(ustring *)sym.data());
        }
        ASSERT (0 && "unhandled constant type");
    }

    return llvm_load_value (llvm_get_pointer (sym), sym.typespec(),
                            deriv, arrayindex, component, cast);
}



llvm::Value *
RuntimeOptimizer::llvm_load_value (llvm::Value *ptr, const TypeSpec &type,
                                   int deriv, llvm::Value *arrayindex,
                                   int component, TypeDesc cast)
{
    if (!ptr)
        return NULL;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t = type.simpletype();
    if (t.arraylen || deriv) {
        int d = deriv * std::max(1,t.arraylen);
        if (arrayindex)
            arrayindex = builder().CreateAdd (arrayindex, llvm_constant(d));
        else
            arrayindex = llvm_constant(d);
        ptr = builder().CreateGEP (ptr, arrayindex);
    }

    // If it's multi-component (triple or matrix), step to the right field
    if (! type.is_closure_based() && t.aggregate > 1)
        ptr = builder().CreateConstGEP2_32 (ptr, 0, component);

    // Now grab the value
    llvm::Value *result = builder().CreateLoad (ptr);

    if (type.is_closure_based())
        return result;

    // Handle int<->float type casting
    if (type.is_floatbased() && cast == TypeDesc::TypeInt)
        result = llvm_float_to_int (result);
    else if (type.is_int() && cast == TypeDesc::TypeFloat)
        result = llvm_int_to_float (result);

    return result;
}



llvm::Value *
RuntimeOptimizer::llvm_load_constant_value (const Symbol& sym, 
                                            int arrayindex, int component,
                                            TypeDesc cast)
{
    ASSERT (sym.is_constant() &&
            "Called llvm_load_constant_value for a non-constant symbol");

    // set array indexing to zero for non-arrays
    if (! sym.typespec().is_array())
        arrayindex = 0;
    ASSERT (arrayindex >= 0 &&
            "Called llvm_load_constant_value with negative array index");

    if (sym.typespec().is_float()) {
        const float *val = (const float *)sym.data();
        if (cast == TypeDesc::TypeInt)
            return llvm_constant ((int)val[arrayindex]);
        else
            return llvm_constant (val[arrayindex]);
    }
    if (sym.typespec().is_int()) {
        const int *val = (const int *)sym.data();
        if (cast == TypeDesc::TypeFloat)
            return llvm_constant ((float)val[arrayindex]);
        else
            return llvm_constant (val[arrayindex]);
    }
    if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
        const float *val = (const float *)sym.data();
        int ncomps = (int) sym.typespec().aggregate();
        return llvm_constant (val[ncomps*arrayindex + component]);
    }
    if (sym.typespec().is_string()) {
        const ustring *val = (const ustring *)sym.data();
        return llvm_constant (val[arrayindex]);
    }

    ASSERT (0 && "unhandled constant type");
    return NULL;
}



llvm::Value *
RuntimeOptimizer::llvm_load_component_value (const Symbol& sym, int deriv,
                                             llvm::Value *component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        ASSERT (sym.typespec().is_floatbased() && 
                "can't ask for derivs of an int");
        return llvm_constant (0.0f);
    }

    // Start with the initial pointer to the value's memory location
    llvm::Value* result = llvm_get_pointer (sym, deriv);
    if (!result)
        return NULL;  // Error

    TypeDesc t = sym.typespec().simpletype();
    ASSERT (t.aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*
    result = llvm_ptr_cast (result, llvm_type_float_ptr());
    result = builder().CreateGEP (result, component);  // get the component

    // Now grab the value
    return builder().CreateLoad (result);
}



llvm::Value *
RuntimeOptimizer::llvm_load_arg (const Symbol& sym, bool derivs)
{
    ASSERT (sym.typespec().is_floatbased());
    if (sym.typespec().is_int() ||
        (sym.typespec().is_float() && !derivs)) {
        // Scalar case
        return llvm_load_value (sym);
    }

    if (derivs && !sym.has_derivs()) {
        // Manufacture-derivs case
        const TypeSpec &t = sym.typespec();
        // Copy the non-deriv values component by component
        llvm::Value *tmpptr = llvm_alloca (t, true);
        for (int c = 0;  c < t.aggregate();  ++c) {
            llvm::Value *v = llvm_load_value (sym, 0, c);
            llvm_store_value (v, tmpptr, t, 0, NULL, c);
        }
        // Zero out the deriv values
        llvm::Value *zero = llvm_constant (0.0f);
        for (int c = 0;  c < t.aggregate();  ++c)
            llvm_store_value (zero, tmpptr, t, 1, NULL, c);
        for (int c = 0;  c < t.aggregate();  ++c)
            llvm_store_value (zero, tmpptr, t, 2, NULL, c);
        return llvm_void_ptr (tmpptr);
    }

    // Regular pointer case
    return llvm_void_ptr (sym);
}



bool
RuntimeOptimizer::llvm_store_value (llvm::Value* new_val, const Symbol& sym,
                                    int deriv, llvm::Value* arrayindex,
                                    int component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    return llvm_store_value (new_val, llvm_get_pointer (sym), sym.typespec(),
                             deriv, arrayindex, component);
}



bool
RuntimeOptimizer::llvm_store_value (llvm::Value* new_val, llvm::Value* dst_ptr,
                                    const TypeSpec &type,
                                    int deriv, llvm::Value* arrayindex,
                                    int component)
{
    if (!dst_ptr)
        return false;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t = type.simpletype();
    if (t.arraylen || deriv) {
        int d = deriv * std::max(1,t.arraylen);
        if (arrayindex)
            arrayindex = builder().CreateAdd (arrayindex, llvm_constant(d));
        else
            arrayindex = llvm_constant(d);
        dst_ptr = builder().CreateGEP (dst_ptr, arrayindex);
    }

    // If it's multi-component (triple or matrix), step to the right field
    if (! type.is_closure_based() && t.aggregate > 1)
        dst_ptr = builder().CreateConstGEP2_32 (dst_ptr, 0, component);

    // Finally, store the value.
    builder().CreateStore (new_val, dst_ptr);
    return true;
}



bool
RuntimeOptimizer::llvm_store_component_value (llvm::Value* new_val,
                                              const Symbol& sym, int deriv,
                                              llvm::Value* component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    // Let llvm_get_pointer do most of the heavy lifting to get us a
    // pointer to where our data lives.
    llvm::Value *result = llvm_get_pointer (sym, deriv);
    if (!result)
        return false;  // Error

    TypeDesc t = sym.typespec().simpletype();
    ASSERT (t.aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*
    result = llvm_ptr_cast (result, llvm_type_float_ptr());
    result = builder().CreateGEP (result, component);  // get the component

    // Finally, store the value.
    builder().CreateStore (new_val, result);
    return true;
}



llvm::Value *
RuntimeOptimizer::layer_run_ptr (int layer)
{
    llvm::Value *layer_run = builder().CreateConstGEP2_32 (groupdata_ptr(), 0, 0);
    return builder().CreateConstGEP2_32 (layer_run, 0, layer);
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (llvm::Value *func,
                                      llvm::Value **args, int nargs)
{
    ASSERT (func);
#if 0
    llvm::outs() << "llvm_call_function " << *func << "\n";
    llvm::outs() << nargs << " args:\n";
    for (int i = 0;  i < nargs;  ++i)
        llvm::outs() << "\t" << *(args[i]) << "\n";
#endif
    //llvm_gen_debug_printf (std::string("start ") + std::string(name));
#if OSL_LLVM_VERSION <= 29
    llvm::Value *r = builder().CreateCall (func, args, args+nargs);
#else
    llvm::Value *r = builder().CreateCall (func, llvm::ArrayRef<llvm::Value *>(args, nargs));
#endif
    //llvm_gen_debug_printf (std::string(" end  ") + std::string(name));
    return r;
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name,
                                      llvm::Value **args, int nargs)
{
    llvm::Function *func = llvm_module()->getFunction (name);
    if (! func)
        std::cerr << "Couldn't find function " << name << "\n";
    return llvm_call_function (func, args, nargs);
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, 
                                      const Symbol **symargs, int nargs,
                                      bool deriv_ptrs)
{
    std::vector<llvm::Value *> valargs;
    valargs.resize ((size_t)nargs);
    for (int i = 0;  i < nargs;  ++i) {
        const Symbol &s = *(symargs[i]);
        if (s.typespec().is_closure())
            valargs[i] = llvm_load_value (s);
        else if (s.typespec().simpletype().aggregate > 1 ||
                 (deriv_ptrs && s.has_derivs()))
            valargs[i] = llvm_void_ptr (s);
        else
            valargs[i] = llvm_load_value (s);
    }
    return llvm_call_function (name, &valargs[0], (int)valargs.size());
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, const Symbol &A,
                                      bool deriv_ptrs)
{
    const Symbol *args[1];
    args[0] = &A;
    return llvm_call_function (name, args, 1, deriv_ptrs);
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, const Symbol &A,
                                      const Symbol &B, bool deriv_ptrs)
{
    const Symbol *args[2];
    args[0] = &A;
    args[1] = &B;
    return llvm_call_function (name, args, 2, deriv_ptrs);
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, const Symbol &A,
                                      const Symbol &B, const Symbol &C,
                                      bool deriv_ptrs)
{
    const Symbol *args[3];
    args[0] = &A;
    args[1] = &B;
    args[2] = &C;
    return llvm_call_function (name, args, 3, deriv_ptrs);
}



void
RuntimeOptimizer::llvm_memset (llvm::Value *ptr, int val,
                               int len, int align)
{
    llvm_memset(ptr, val, llvm_constant(len), align);
}



void
RuntimeOptimizer::llvm_memset (llvm::Value *ptr, int val,
                               llvm::Value *len, int align)
{
    // memset with i32 len
    // and with an i8 pointer (dst) for LLVM-2.8
    llvm::Type* types[] = {
        (llvm::Type *) llvm::PointerType::get(llvm::Type::getInt8Ty(llvm_context()), 0),
        (llvm::Type *) llvm::Type::getInt32Ty(llvm_context())
    };

#if OSL_LLVM_VERSION <= 29
    llvm::Function* func = llvm::Intrinsic::getDeclaration (llvm_module(),
        llvm::Intrinsic::memset,
        (const llvm::Type**) types,
        sizeof(types)/sizeof(llvm::Type*));
#else
    llvm::Function* func = llvm::Intrinsic::getDeclaration (llvm_module(),
        llvm::Intrinsic::memset,
        llvm::ArrayRef<llvm::Type *>(types, sizeof(types)/sizeof(llvm::Type*)));
#endif

    // NOTE(boulos): llvm_constant(0) would return an i32
    // version of 0, but we need the i8 version. If we make an
    // ::llvm_constant(char val) though then we'll get ambiguity
    // everywhere.
    llvm::Value* fill_val = llvm::ConstantInt::get (llvm_context(),
                                                    llvm::APInt(8, val));
    // Non-volatile (allow optimizer to move it around as it wishes
    // and even remove it if it can prove it's useless)
    builder().CreateCall5 (func, ptr, fill_val, len, llvm_constant(align),
                           llvm_constant_bool(false));
}



void
RuntimeOptimizer::llvm_memcpy (llvm::Value *dst, llvm::Value *src,
                               int len, int align)
{
    // i32 len
    // and with i8 pointers (dst and src) for LLVM-2.8
    llvm::Type* types[] = {
        (llvm::Type *) llvm::PointerType::get(llvm::Type::getInt8Ty(llvm_context()), 0),
        (llvm::Type *) llvm::PointerType::get(llvm::Type::getInt8Ty(llvm_context()), 0),
        (llvm::Type *) llvm::Type::getInt32Ty(llvm_context())
    };

#if OSL_LLVM_VERSION <= 29
    llvm::Function* func = llvm::Intrinsic::getDeclaration (llvm_module(),
        llvm::Intrinsic::memcpy,
        (const llvm::Type**) types,
        sizeof(types) / sizeof(llvm::Type*));
#else
    llvm::Function* func = llvm::Intrinsic::getDeclaration (llvm_module(),
        llvm::Intrinsic::memcpy,
        llvm::ArrayRef<llvm::Type *>(types, sizeof(types)/sizeof(llvm::Type*)));
#endif

    // Non-volatile (allow optimizer to move it around as it wishes
    // and even remove it if it can prove it's useless)
    builder().CreateCall5 (func, dst, src,
                           llvm_constant(len), llvm_constant(align), llvm_constant_bool(false));
}



/// Convert a float llvm value to an integer.
///
llvm::Value *
RuntimeOptimizer::llvm_float_to_int (llvm::Value* fval)
{
    return builder().CreateFPToSI(fval, llvm_type_int());
}



/// Convert an integer llvm value to a float.
///
llvm::Value *
RuntimeOptimizer::llvm_int_to_float (llvm::Value* ival)
{
    return builder().CreateSIToFP(ival, llvm_type_float());
}



llvm::Value *
RuntimeOptimizer::llvm_make_safe_div (TypeDesc type,
                                      llvm::Value *a, llvm::Value *b)
{
    if (type.basetype == TypeDesc::FLOAT) {
        llvm::Value *div = builder().CreateFDiv (a, b);
        llvm::Value *zero = llvm_constant (0.0f);
        llvm::Value *iszero = builder().CreateFCmpOEQ (b, zero);
        return builder().CreateSelect (iszero, zero, div);
    } else {
        llvm::Value *div = builder().CreateSDiv (a, b);
        llvm::Value *zero = llvm_constant (0);
        llvm::Value *iszero = builder().CreateICmpEQ (b, zero);
        return builder().CreateSelect (iszero, zero, div);
    }
}



llvm::Value *
RuntimeOptimizer::llvm_make_safe_mod (TypeDesc type,
                                      llvm::Value *a, llvm::Value *b)
{
    if (type.basetype == TypeDesc::FLOAT) {
        llvm::Value *mod = builder().CreateFRem (a, b);
        llvm::Value *zero = llvm_constant (0.0f);
        llvm::Value *iszero = builder().CreateFCmpOEQ (b, zero);
        return builder().CreateSelect (iszero, zero, mod);
    } else {
        llvm::Value *mod = builder().CreateSRem (a, b);
        llvm::Value *zero = llvm_constant (0);
        llvm::Value *iszero = builder().CreateICmpEQ (b, zero);
        return builder().CreateSelect (iszero, zero, mod);
    }
}



bool
RuntimeOptimizer::llvm_assign_impl (Symbol &Result, Symbol &Src,
                                    int arrayindex)
{
    ASSERT (! Result.typespec().is_structure());
    ASSERT (! Src.typespec().is_structure());

    const TypeSpec &result_t (Result.typespec());
    const TypeSpec &src_t (Src.typespec());

    llvm::Value *arrind = arrayindex >= 0 ? llvm_constant (arrayindex) : NULL;

    if (Result.typespec().is_closure_based() || Src.typespec().is_closure_based()) {
        if (Src.typespec().is_closure_based()) {
            llvm::Value *srcval = llvm_load_value (Src, 0, arrind, 0);
            llvm_store_value (srcval, Result, 0, arrind, 0);
        } else {
            llvm::Value *null = llvm_constant_ptr(NULL, llvm_type_void_ptr());
            llvm_store_value (null, Result, 0, arrind, 0);
        }
        return true;
    }

    if (Result.typespec().is_matrix() && Src.typespec().is_int_or_float()) {
        // Handle m=f, m=i separately
        llvm::Value *src = llvm_load_value (Src, 0, arrind, 0, TypeDesc::FLOAT /*cast*/);
        // m=f sets the diagonal components to f, the others to zero
        llvm::Value *zero = llvm_constant (0.0f);
        for (int i = 0;  i < 4;  ++i)
            for (int j = 0;  j < 4;  ++j)
                llvm_store_value (i==j ? src : zero, Result, 0, arrind, i*4+j);
        llvm_zero_derivs (Result);  // matrices don't have derivs currently
        return true;
    }

    // Copying of entire arrays
    if (result_t.is_array() && src_t.is_array() && arrayindex == -1) {
        ASSERT (assignable(result_t.elementtype(), src_t.elementtype()) &&
                result_t.arraylength() == src_t.arraylength());
        llvm::Value *resultptr = llvm_void_ptr (Result);
        llvm::Value *srcptr = llvm_void_ptr (Src);
        int len = Result.size();
        int align = result_t.is_closure_based() ? (int)sizeof(void*) :
                                       (int)result_t.simpletype().basesize();
        if (Result.has_derivs() && Src.has_derivs()) {
            llvm_memcpy (resultptr, srcptr, 3*len, align);
        } else {
            llvm_memcpy (resultptr, srcptr, len, align);
            if (Result.has_derivs())
                llvm_zero_derivs (Result);
        }
        return true;
    }

    // The following code handles f=f, f=i, v=v, v=f, v=i, m=m, s=s.
    // Remember that llvm_load_value will automatically convert scalar->triple.
    TypeDesc rt = Result.typespec().simpletype();
    TypeDesc basetype = TypeDesc::BASETYPE(rt.basetype);
    int num_components = rt.aggregate;
    for (int i = 0; i < num_components; ++i) {
        llvm::Value* src_val = Src.is_constant()
            ? llvm_load_constant_value (Src, arrayindex, i, basetype)
            : llvm_load_value (Src, 0, arrind, i, basetype);
        if (!src_val)
            return false;
        llvm_store_value (src_val, Result, 0, arrind, i);
    }

    // Handle derivatives
    if (Result.has_derivs()) {
        if (Src.has_derivs()) {
            // src and result both have derivs -- copy them
            for (int d = 1;  d <= 2;  ++d) {
                for (int i = 0; i < num_components; ++i) {
                    llvm::Value* val = llvm_load_value (Src, d, arrind, i);
                    llvm_store_value (val, Result, d, arrind, i);
                }
            }
        } else {
            // Result wants derivs but src didn't have them -- zero them
            llvm_zero_derivs (Result);
        }
    }
    return true;
}




}; // namespace pvt
}; // namespace osl

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
