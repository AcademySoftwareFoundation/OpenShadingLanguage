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


#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>

#include "oslexec_pvt.h"
#include "backendllvm.h"

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace pvt {


#ifdef OSL_SPI
static void
check_cwd (ShadingSystemImpl &shadingsys)
{
    std::string err;
    char pathname[1024] = { "" };
    if (! getcwd (pathname, sizeof(pathname)-1)) {
        int e = errno;
        err += Strutil::format ("Failed getcwd(), errno is %d: %s\n",
                                errno, pathname);
        if (e == EACCES || e == ENOENT) {
            err += "Read/search permission problem or dir does not exist.\n";
            const char *pwdenv = getenv ("PWD");
            if (! pwdenv) {
                err += "$PWD is not even found in the environment.\n";
            } else {
                err += Strutil::format ("$PWD is \"%s\"\n", pwdenv);
                err += Strutil::format ("That %s.\n",
                          OIIO::Filesystem::exists(pwdenv) ? "exists" : "does NOT exist");
                err += Strutil::format ("That %s a directory.\n",
                          OIIO::Filesystem::is_directory(pwdenv) ? "is" : "is NOT");
                std::vector<std::string> pieces;
                Strutil::split (pwdenv, pieces, "/");
                std::string p;
                for (size_t i = 0;  i < pieces.size();  ++i) {
                    if (! pieces[i].size())
                        continue;
                    p += "/";
                    p += pieces[i];
                    err += Strutil::format ("  %s : %s and is%s a directory.\n", p,
                        OIIO::Filesystem::exists(p) ? "exists" : "does NOT exist",
                        OIIO::Filesystem::is_directory(p) ? "" : " NOT");
                }
            }
        }
    }
    if (err.size())
        shadingsys.error (err);
}
#endif



BackendLLVM::BackendLLVM (ShadingSystemImpl &shadingsys,
                          ShaderGroup &group, ShadingContext *ctx)
    : OSOProcessorBase (shadingsys, group, ctx),
      ll(llvm_debug()),
      m_stat_total_llvm_time(0), m_stat_llvm_setup_time(0),
      m_stat_llvm_irgen_time(0), m_stat_llvm_opt_time(0),
      m_stat_llvm_jit_time(0)
{
#ifdef OSL_SPI
    // Temporary (I hope) check to diagnose an intermittent failure of
    // getcwd inside LLVM. Oy.
    check_cwd (shadingsys);
#endif
    int mcjit = 0;
    if (shadingsys.getattribute ("llvm_mcjit", TypeDesc::INT, &mcjit))
        ll.mcjit (mcjit);
}



BackendLLVM::~BackendLLVM ()
{
}



int
BackendLLVM::llvm_debug() const
{
    if (shadingsys().llvm_debug() == 0)
        return 0;
    if (shadingsys().debug_groupname() &&
        shadingsys().debug_groupname() != group().name())
        return 0;
    if (inst() && shadingsys().debug_layername() &&
        shadingsys().debug_layername() != inst()->layername())
        return 0;
    return shadingsys().llvm_debug();
}



void
BackendLLVM::set_inst (int layer)
{
    OSOProcessorBase::set_inst (layer);  // parent does the heavy lifting
    ll.debug (llvm_debug());
}



llvm::Type *
BackendLLVM::llvm_pass_type (const TypeSpec &typespec)
{
    if (typespec.is_closure_based())
        return (llvm::Type *) ll.type_void_ptr();
    TypeDesc t = typespec.simpletype().elementtype();
    llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = ll.type_float();
    else if (t == TypeDesc::INT)
        lt = ll.type_int();
    else if (t == TypeDesc::STRING)
        lt = (llvm::Type *) ll.type_string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = (llvm::Type *) ll.type_void_ptr(); //llvm_type_triple_ptr();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = (llvm::Type *) ll.type_void_ptr(); //llvm_type_matrix_ptr();
    else if (t == TypeDesc::NONE)
        lt = ll.type_void();
    else if (t == TypeDesc::PTR)
        lt = (llvm::Type *) ll.type_void_ptr();
    else if (t == TypeDesc::LONGLONG)
        lt = ll.type_longlong();
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
BackendLLVM::llvm_assign_zero (const Symbol &sym)
{
    // Just memset the whole thing to zero, let LLVM sort it out.
    // This even works for closures.
    int len;
    if (sym.typespec().is_closure_based())
        len = sizeof(void *) * sym.typespec().numelements();
    else
        len = sym.derivsize();
    // N.B. derivsize() includes derivs, if there are any
    size_t align = sym.typespec().is_closure_based() ? sizeof(void*) :
                         sym.typespec().simpletype().basesize();
    ll.op_memset (llvm_void_ptr(sym), 0, len, (int)align);
}



void
BackendLLVM::llvm_zero_derivs (const Symbol &sym)
{
    if (sym.typespec().is_closure_based())
        return; // Closures don't have derivs
    // Just memset the derivs to zero, let LLVM sort it out.
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_floatbased()) {
        int len = sym.size();
        size_t align = sym.typespec().simpletype().basesize();
        ll.op_memset (llvm_void_ptr(sym,1), /* point to start of x deriv */
                      0, 2*len /* size of both derivs */, (int)align);
    }
}



void
BackendLLVM::llvm_zero_derivs (const Symbol &sym, llvm::Value *count)
{
    if (sym.typespec().is_closure_based())
        return; // Closures don't have derivs
    // Same thing as the above version but with just the first count derivs
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_floatbased()) {
        size_t esize = sym.typespec().simpletype().elementsize();
        size_t align = sym.typespec().simpletype().basesize();
        count = ll.op_mul (count, ll.constant((int)esize));
        ll.op_memset (llvm_void_ptr(sym,1), 0, count, (int)align); // X derivs
        ll.op_memset (llvm_void_ptr(sym,2), 0, count, (int)align); // Y derivs
    }
}



int
BackendLLVM::ShaderGlobalNameToIndex (ustring name)
{
    // N.B. The order of names in this table MUST exactly match the
    // ShaderGlobals struct in oslexec.h, as well as the llvm 'sg' type
    // defined in llvm_type_sg().
    static ustring fields[] = {
        Strings::P, ustring("_dPdz"), Strings::I, Strings::N, Strings::Ng,
        Strings::u, Strings::v, Strings::dPdu, Strings::dPdv,
        Strings::time, Strings::dtime, Strings::dPdtime, Strings::Ps,
        ustring("renderstate"), ustring("tracedata"), ustring("objdata"),
        ustring("shadingcontext"), ustring("renderer"),
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
BackendLLVM::llvm_global_symbol_ptr (ustring name)
{
    // Special case for globals -- they live in the ShaderGlobals struct,
    // we use the name of the global to find the index of the field within
    // the ShaderGlobals struct.
    int sg_index = ShaderGlobalNameToIndex (name);
    ASSERT (sg_index >= 0);
    return ll.void_ptr (ll.GEP (sg_ptr(), 0, sg_index));
}



llvm::Value *
BackendLLVM::getLLVMSymbolBase (const Symbol &sym)
{
    Symbol* dealiased = sym.dealias();

    if (sym.symtype() == SymTypeGlobal) {
        llvm::Value *result = llvm_global_symbol_ptr (sym.name());
        ASSERT (result);
        result = ll.ptr_to_cast (result, llvm_type(sym.typespec().elementtype()));
        return result;
    }

    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        // Special case for params -- they live in the group data
        int fieldnum = m_param_order_map[&sym];
        return groupdata_field_ptr (fieldnum, sym.typespec().elementtype().simpletype());
    }

    std::string mangled_name = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find (mangled_name);
    if (map_iter == named_values().end()) {
        shadingcontext()->error ("Couldn't find symbol '%s' (unmangled = '%s'). Did you forget to allocate it?",
                            mangled_name.c_str(), dealiased->name().c_str());
        return 0;
    }
    return (llvm::Value*) map_iter->second;
}



llvm::Value *
BackendLLVM::llvm_alloca (const TypeSpec &type, bool derivs,
                          const std::string &name)
{
    TypeDesc t = llvm_typedesc (type);
    int n = derivs ? 3 : 1;
    m_llvm_local_mem += t.size() * n;
    return ll.op_alloca (t, n, name);
}



llvm::Value *
BackendLLVM::getOrAllocateLLVMSymbol (const Symbol& sym)
{
    DASSERT ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp ||
              sym.symtype() == SymTypeConst)
             && "getOrAllocateLLVMSymbol should only be for local, tmp, const");
    Symbol* dealiased = sym.dealias();
    std::string mangled_name = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);

    if (map_iter == named_values().end()) {
        llvm::Value* a = llvm_alloca (sym.typespec(), sym.has_derivs(), mangled_name);
        named_values()[mangled_name] = a;
        return a;
    }
    return map_iter->second;
}



llvm::Value *
BackendLLVM::llvm_get_pointer (const Symbol& sym, int deriv,
                               llvm::Value *arrayindex)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Return NULL for request for pointer to derivs that don't exist
        return ll.ptr_cast (ll.void_ptr_null(),
                            ll.type_ptr (llvm_type(sym.typespec().elementtype())));
    }

    llvm::Value *result = NULL;
    if (sym.symtype() == SymTypeConst) {
        // For constants, start with *OUR* pointer to the constant values.
        result = ll.ptr_cast (ll.constant_ptr (sym.data()),
                              ll.type_ptr (llvm_type(sym.typespec().elementtype())));

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
            arrayindex = ll.op_add (arrayindex, ll.constant(d));
        else
            arrayindex = ll.constant(d);
        result = ll.GEP (result, arrayindex);
    }

    return result;
}



llvm::Value *
BackendLLVM::llvm_load_value (const Symbol& sym, int deriv,
                                   llvm::Value *arrayindex, int component,
                                   TypeDesc cast)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        return ll.constant (0.0f);
    }

    // arrayindex should be non-NULL if and only if sym is an array
    ASSERT (sym.typespec().is_array() == (arrayindex != NULL));

    if (sym.is_constant() && !sym.typespec().is_array() && !arrayindex) {
        // Shortcut for simple constants
        if (sym.typespec().is_float()) {
            if (cast == TypeDesc::TypeInt)
                return ll.constant ((int)*(float *)sym.data());
            else
                return ll.constant (*(float *)sym.data());
        }
        if (sym.typespec().is_int()) {
            if (cast == TypeDesc::TypeFloat)
                return ll.constant ((float)*(int *)sym.data());
            else
                return ll.constant (*(int *)sym.data());
        }
        if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
            return ll.constant (((float *)sym.data())[component]);
        }
        if (sym.typespec().is_string()) {
            return ll.constant (*(ustring *)sym.data());
        }
        ASSERT (0 && "unhandled constant type");
    }

    return llvm_load_value (llvm_get_pointer (sym), sym.typespec(),
                            deriv, arrayindex, component, cast);
}



llvm::Value *
BackendLLVM::llvm_load_value (llvm::Value *ptr, const TypeSpec &type,
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
            arrayindex = ll.op_add (arrayindex, ll.constant(d));
        else
            arrayindex = ll.constant(d);
        ptr = ll.GEP (ptr, arrayindex);
    }

    // If it's multi-component (triple or matrix), step to the right field
    if (! type.is_closure_based() && t.aggregate > 1)
        ptr = ll.GEP (ptr, 0, component);

    // Now grab the value
    llvm::Value *result = ll.op_load (ptr);

    if (type.is_closure_based())
        return result;

    // Handle int<->float type casting
    if (type.is_floatbased() && cast == TypeDesc::TypeInt)
        result = ll.op_float_to_int (result);
    else if (type.is_int() && cast == TypeDesc::TypeFloat)
        result = ll.op_int_to_float (result);

    return result;
}



llvm::Value *
BackendLLVM::llvm_load_constant_value (const Symbol& sym, 
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
            return ll.constant ((int)val[arrayindex]);
        else
            return ll.constant (val[arrayindex]);
    }
    if (sym.typespec().is_int()) {
        const int *val = (const int *)sym.data();
        if (cast == TypeDesc::TypeFloat)
            return ll.constant ((float)val[arrayindex]);
        else
            return ll.constant (val[arrayindex]);
    }
    if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
        const float *val = (const float *)sym.data();
        int ncomps = (int) sym.typespec().aggregate();
        return ll.constant (val[ncomps*arrayindex + component]);
    }
    if (sym.typespec().is_string()) {
        const ustring *val = (const ustring *)sym.data();
        return ll.constant (val[arrayindex]);
    }

    ASSERT (0 && "unhandled constant type");
    return NULL;
}



llvm::Value *
BackendLLVM::llvm_load_component_value (const Symbol& sym, int deriv,
                                             llvm::Value *component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        ASSERT (sym.typespec().is_floatbased() && 
                "can't ask for derivs of an int");
        return ll.constant (0.0f);
    }

    // Start with the initial pointer to the value's memory location
    llvm::Value* result = llvm_get_pointer (sym, deriv);
    if (!result)
        return NULL;  // Error

    TypeDesc t = sym.typespec().simpletype();
    ASSERT (t.aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*
    result = ll.ptr_cast (result, ll.type_float_ptr());
    result = ll.GEP (result, component);  // get the component

    // Now grab the value
    return ll.op_load (result);
}



llvm::Value *
BackendLLVM::llvm_load_arg (const Symbol& sym, bool derivs)
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
        llvm::Value *zero = ll.constant (0.0f);
        for (int c = 0;  c < t.aggregate();  ++c)
            llvm_store_value (zero, tmpptr, t, 1, NULL, c);
        for (int c = 0;  c < t.aggregate();  ++c)
            llvm_store_value (zero, tmpptr, t, 2, NULL, c);
        return ll.void_ptr (tmpptr);
    }

    // Regular pointer case
    return llvm_void_ptr (sym);
}



bool
BackendLLVM::llvm_store_value (llvm::Value* new_val, const Symbol& sym,
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
BackendLLVM::llvm_store_value (llvm::Value* new_val, llvm::Value* dst_ptr,
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
            arrayindex = ll.op_add (arrayindex, ll.constant(d));
        else
            arrayindex = ll.constant(d);
        dst_ptr = ll.GEP (dst_ptr, arrayindex);
    }

    // If it's multi-component (triple or matrix), step to the right field
    if (! type.is_closure_based() && t.aggregate > 1)
        dst_ptr = ll.GEP (dst_ptr, 0, component);

    // Finally, store the value.
    ll.op_store (new_val, dst_ptr);
    return true;
}



bool
BackendLLVM::llvm_store_component_value (llvm::Value* new_val,
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
    result = ll.ptr_cast (result, ll.type_float_ptr());
    result = ll.GEP (result, component);  // get the component

    // Finally, store the value.
    ll.op_store (new_val, result);
    return true;
}



llvm::Value *
BackendLLVM::groupdata_field_ref (int fieldnum)
{
    return ll.GEP (groupdata_ptr(), 0, fieldnum);
}


llvm::Value *
BackendLLVM::groupdata_field_ptr (int fieldnum, TypeDesc type)
{
    llvm::Value *result = ll.void_ptr (groupdata_field_ref (fieldnum));
    if (type != TypeDesc::UNKNOWN)
        result = ll.ptr_to_cast (result, llvm_type(type));
    return result;
}


llvm::Value *
BackendLLVM::layer_run_ref (int layer)
{
    int fieldnum = 0; // field 0 is the layer_run array
    llvm::Value *layer_run = groupdata_field_ref (fieldnum);
    return ll.GEP (layer_run, 0, layer);
}



llvm::Value *
BackendLLVM::userdata_initialized_ref (int userdata_index)
{
    int fieldnum = 1; // field 1 is the userdata_initialized array
    llvm::Value *userdata_initiazlied = groupdata_field_ref (fieldnum);
    return ll.GEP (userdata_initiazlied, 0, userdata_index);
}



llvm::Value *
BackendLLVM::llvm_call_function (const char *name, 
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
    return ll.call_function (name, (valargs.size())? &valargs[0]: NULL,
                             (int)valargs.size());
}



llvm::Value *
BackendLLVM::llvm_call_function (const char *name, const Symbol &A,
                                 bool deriv_ptrs)
{
    const Symbol *args[1];
    args[0] = &A;
    return llvm_call_function (name, args, 1, deriv_ptrs);
}



llvm::Value *
BackendLLVM::llvm_call_function (const char *name, const Symbol &A,
                                 const Symbol &B, bool deriv_ptrs)
{
    const Symbol *args[2];
    args[0] = &A;
    args[1] = &B;
    return llvm_call_function (name, args, 2, deriv_ptrs);
}



llvm::Value *
BackendLLVM::llvm_call_function (const char *name, const Symbol &A,
                                 const Symbol &B, const Symbol &C,
                                 bool deriv_ptrs)
{
    const Symbol *args[3];
    args[0] = &A;
    args[1] = &B;
    args[2] = &C;
    return llvm_call_function (name, args, 3, deriv_ptrs);
}



llvm::Value *
BackendLLVM::llvm_test_nonzero (Symbol &val, bool test_derivs)
{
    const TypeSpec &ts (val.typespec());
    ASSERT (! ts.is_array() && ! ts.is_closure() && ! ts.is_string());
    TypeDesc t = ts.simpletype();

    // Handle int case -- guaranteed no derivs, no multi-component
    if (t == TypeDesc::TypeInt)
        return ll.op_ne (llvm_load_value(val), ll.constant(0));

    // float-based
    int ncomps = t.aggregate;
    int nderivs = (test_derivs && val.has_derivs()) ? 3 : 1;
    llvm::Value *isnonzero = NULL;
    for (int d = 0;  d < nderivs;  ++d) {
        for (int c = 0;  c < ncomps;  ++c) {
            llvm::Value *v = llvm_load_value (val, d, c);
            llvm::Value *nz = ll.op_ne (v, ll.constant(0.0f), true);
            if (isnonzero)  // multi-component/deriv: OR with running result
                isnonzero = ll.op_or (nz, isnonzero);
            else
                isnonzero = nz;
        }
    }
    return isnonzero;
}



bool
BackendLLVM::llvm_assign_impl (Symbol &Result, Symbol &Src,
                                    int arrayindex)
{
    ASSERT (! Result.typespec().is_structure());
    ASSERT (! Src.typespec().is_structure());

    const TypeSpec &result_t (Result.typespec());
    const TypeSpec &src_t (Src.typespec());

    llvm::Value *arrind = arrayindex >= 0 ? ll.constant (arrayindex) : NULL;

    if (Result.typespec().is_closure_based() || Src.typespec().is_closure_based()) {
        if (Src.typespec().is_closure_based()) {
            llvm::Value *srcval = llvm_load_value (Src, 0, arrind, 0);
            llvm_store_value (srcval, Result, 0, arrind, 0);
        } else {
            llvm::Value *null = ll.constant_ptr(NULL, ll.type_void_ptr());
            llvm_store_value (null, Result, 0, arrind, 0);
        }
        return true;
    }

    if (Result.typespec().is_matrix() && Src.typespec().is_int_or_float()) {
        // Handle m=f, m=i separately
        llvm::Value *src = llvm_load_value (Src, 0, arrind, 0, TypeDesc::FLOAT /*cast*/);
        // m=f sets the diagonal components to f, the others to zero
        llvm::Value *zero = ll.constant (0.0f);
        for (int i = 0;  i < 4;  ++i)
            for (int j = 0;  j < 4;  ++j)
                llvm_store_value (i==j ? src : zero, Result, 0, arrind, i*4+j);
        llvm_zero_derivs (Result);  // matrices don't have derivs currently
        return true;
    }

    // Copying of entire arrays.  It's ok if the array lengths don't match,
    // it will only copy up to the length of the smaller one.  The compiler
    // will ensure they are the same size, except for certain cases where
    // the size difference is intended (by the optimizer).
    if (result_t.is_array() && src_t.is_array() && arrayindex == -1) {
        ASSERT (assignable(result_t.elementtype(), src_t.elementtype()));
        llvm::Value *resultptr = llvm_void_ptr (Result);
        llvm::Value *srcptr = llvm_void_ptr (Src);
        int len = std::min (Result.size(), Src.size());
        int align = result_t.is_closure_based() ? (int)sizeof(void*) :
                                       (int)result_t.simpletype().basesize();
        if (Result.has_derivs() && Src.has_derivs()) {
            ll.op_memcpy (resultptr, srcptr, 3*len, align);
        } else {
            ll.op_memcpy (resultptr, srcptr, len, align);
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
OSL_NAMESPACE_EXIT
