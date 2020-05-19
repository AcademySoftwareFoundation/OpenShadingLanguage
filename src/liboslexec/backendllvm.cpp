// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>

#ifdef OSL_USE_OPTIX
#include <optix.h>
#endif

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
        err += Strutil::sprintf ("Failed getcwd(), errno is %d: %s\n",
                                errno, pathname);
        if (e == EACCES || e == ENOENT) {
            err += "Read/search permission problem or dir does not exist.\n";
            const char *pwdenv = getenv ("PWD");
            if (! pwdenv) {
                err += "$PWD is not even found in the environment.\n";
            } else {
                err += Strutil::sprintf ("$PWD is \"%s\"\n", pwdenv);
                err += Strutil::sprintf ("That %s.\n",
                          OIIO::Filesystem::exists(pwdenv) ? "exists" : "does NOT exist");
                err += Strutil::sprintf ("That %s a directory.\n",
                          OIIO::Filesystem::is_directory(pwdenv) ? "is" : "is NOT");
                std::vector<std::string> pieces;
                Strutil::split (pwdenv, pieces, "/");
                std::string p;
                for (size_t i = 0;  i < pieces.size();  ++i) {
                    if (! pieces[i].size())
                        continue;
                    p += "/";
                    p += pieces[i];
                    err += Strutil::sprintf ("  %s : %s and is%s a directory.\n", p,
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
    m_use_optix = shadingsys.renderer()->supports ("OptiX");
}



BackendLLVM::~BackendLLVM ()
{
}



int
BackendLLVM::llvm_debug() const
{
    if (shadingsys().llvm_debug() == 0)
        return 0;
    if (!shadingsys().debug_groupname().empty() &&
        shadingsys().debug_groupname() != group().name())
        return 0;
    if (inst() && !shadingsys().debug_layername().empty() &&
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
        OSL_ASSERT_MSG (0, "not handling %s type yet", typespec.c_str());
    }
    if (t.arraylen) {
        OSL_ASSERT (0 && "should never pass an array directly as a parameter");
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

namespace
{
    // N.B. The order of names in this table MUST exactly match the
    // ShaderGlobals struct in oslexec.h, as well as the llvm 'sg' type
    // defined in llvm_type_sg().
    static ustring fields[] = {
        ustring("P"), ustring("_dPdz"), ustring("I"),
        ustring("N"), ustring("Ng"),
        ustring("u"), ustring("v"), ustring("dPdu"), ustring("dPdv"),
        ustring("time"), ustring("dtime"), ustring("dPdtime"), ustring("Ps"),
        ustring("renderstate"), ustring("tracedata"), ustring("objdata"),
        ustring("shadingcontext"), ustring("renderer"),
        ustring("object2common"), ustring("shader2common"),
        ustring("Ci"),
        ustring("surfacearea"), ustring("raytype"),
        ustring("flipHandedness"), ustring("backfacing")
    };
}

int
BackendLLVM::ShaderGlobalNameToIndex (ustring name)
{
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
    OSL_ASSERT (sg_index >= 0);
    return ll.void_ptr (ll.GEP (sg_ptr(), 0, sg_index));
}



llvm::Value *
BackendLLVM::getLLVMSymbolBase (const Symbol &sym)
{
    Symbol* dealiased = sym.dealias();

    if (sym.symtype() == SymTypeGlobal) {
        llvm::Value *result = llvm_global_symbol_ptr (sym.name());
        OSL_ASSERT (result);
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
        shadingcontext()->errorf("Couldn't find symbol '%s' (unmangled = '%s'). Did you forget to allocate it?",
                                 mangled_name, dealiased->name());
        return 0;
    }
    return (llvm::Value*) map_iter->second;
}



llvm::Value *
BackendLLVM::llvm_alloca (const TypeSpec &type, bool derivs,
                          const std::string &name, int align)
{
    TypeDesc t = llvm_typedesc (type);
    int n = derivs ? 3 : 1;
    m_llvm_local_mem += t.size() * n;
    return ll.op_alloca (t, n, name, align);
}



llvm::Value *
BackendLLVM::getOrAllocateLLVMSymbol (const Symbol& sym)
{
    OSL_DASSERT((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp ||
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



llvm::Value*
BackendLLVM::addCUDAVariable(const std::string& name, int size, int alignment,
                             const void* data, TypeDesc type)
{
    OSL_ASSERT (use_optix() && "This function is only supposed to be used with OptiX!");

    llvm::Constant* constant = nullptr;

    if (type == TypeDesc::TypeFloat) {
        constant = llvm::ConstantFP::get (
            llvm::Type::getFloatTy (ll.module()->getContext()), *(float*) data);
    }
    else if (type == TypeDesc::TypeInt) {
        constant = llvm::ConstantInt::get (
            llvm::Type::getInt32Ty (ll.module()->getContext()), *(int*) data);
    }
    else if (type == TypeDesc::TypeString) {
        // Register the string with the OptiX renderer. The renderer will add
        // the string to a global table and create an OptiX variable to hold the
        // char*.
#if (OPTIX_VERSION < 70000)
        shadingsys().renderer()->register_string (((ustring*)data)->string(), name);

        // Leave the variable uninitialized to prevent raw pointers from
        // appearing in the generated code. The OptiX renderer will set the
        // variable to the string address before the kernel is launched.
        constant = llvm::ConstantInt::get (
            llvm::Type::getInt64Ty (ll.module()->getContext()), 0);
#else
        // TODO:  don't perform variable assignment in generated PTX code
        int64_t addr = shadingsys().renderer()->register_string (((ustring*)data)->string(), name);
        constant = llvm::ConstantInt::get (
            llvm::Type::getInt64Ty (ll.module()->getContext()), addr);
#endif
        m_varname_map [name] = ((ustring*)data)->string();
    }
    else {
        // Handle unspecified types as generic byte arrays
        llvm::ArrayRef<uint8_t> arr_ref ((uint8_t*)data, size);
        constant = llvm::ConstantDataArray::get (ll.module()->getContext(), arr_ref);
    }

    llvm::GlobalVariable* g_var = reinterpret_cast<llvm::GlobalVariable*>(
        ll.module()->getOrInsertGlobal (name, constant->getType()));

    OSL_DASSERT (g_var && "Unable to create GlobalVariable");

#if OSL_LLVM_VERSION >= 100
    g_var->setAlignment  (llvm::MaybeAlign(alignment));
#else
    g_var->setAlignment  (alignment);
#endif
    g_var->setLinkage    (llvm::GlobalValue::ExternalLinkage);
    g_var->setVisibility (llvm::GlobalValue::DefaultVisibility);
    g_var->setInitializer(constant);
#if (OPTIX_VERSION >= 70000)
    if (type == TypeDesc::TypeString)
        g_var->setConstant(true);
#endif
    m_const_map[name] = g_var;

    return g_var;
}



void
BackendLLVM::createOptixMetadata (const std::string& name, const Symbol& sym)
{
    // Create additional variables with the semantic information needed by OptiX
    // to access the global variable created above.
    //
    // There is no need to retain pointers to these variables, since they are not
    // accessed during execution. They are only used internally by OptiX.
    //
    // Refer to the OptiX API documentation and optix_defines.h in the OptiX SDK
    // for more information.

    OSL_ASSERT (use_optix() && "This function is only supported when using OptiX!");

    auto mangle_name = [](const std::string& name, const std::string& prefix) {
        return OIIO::Strutil::sprintf ("_ZN%drti_internal_%s%d%sE",
                                      prefix.size()+13, prefix, name.size(), name);
    };

    std::string optix_type;
    const TypeDesc type = sym.typespec().simpletype();
    if (! sym.typespec().is_array()) {
        optix_type =
            // Documented built-in types
            (type == TypeDesc::TypeInt   ) ? "int"      :
            (type == TypeDesc::TypeFloat ) ? "float"    :
            (type == TypeDesc::TypePoint ) ? "float3"   :
            (type == TypeDesc::TypeVector) ? "float3"   :
            (type == TypeDesc::TypeNormal) ? "float3"   :
            (type == TypeDesc::TypeColor ) ? "float3"   :
            (type == TypeDesc::TypeMatrix) ? "matrix"   :
            (type == TypeDesc::TypeString) ? "uint64_t" :
            // Catch-all for types that fall through, if there are any.
            type.c_str();

        // NB: TypeMatrix is assumed to be 4x4 and will be treated by OptiX as a
        //     user datatype (i.e., a generic struct).
    }
    else {
        // OptiX should handle int and float vectors between 2 and 4 dimensions
        // with no problem. Larger arrays, or arrays of other element types,
        // will be treated as a user datatype and will not work natively with
        // OptiX's variable mechanism.
        optix_type = OIIO::Strutil::sprintf ("%s%d", sym.typespec().elementtype().c_str(),
                                             sym.typespec().arraylength());
    }

    struct rti_typeinfo {
        unsigned int kind = 0x796152; // _OPTIX_VARIABLE
        unsigned int size;
    } type_info;
    type_info.size = sym.size();

    int  type_enum = 0x1337;          // _OPTIX_TYPE_ENUM_UNKNOWN
    char zero      = 0;

    addCUDAVariable (mangle_name (name, "typeinfo"  ), 8, 4, &type_info);
    addCUDAVariable (mangle_name (name, "typename"  ), optix_type.size() + 1, 16, optix_type.data());
    addCUDAVariable (mangle_name (name, "typeenum"  ), 4, 4, &type_enum, TypeDesc::TypeInt);
    addCUDAVariable (mangle_name (name, "semantic"  ), 1, 1, &zero);
    addCUDAVariable (mangle_name (name, "annotation"), 1, 1, &zero);
}



llvm::Value *
BackendLLVM::getOrAllocateCUDAVariable (const Symbol& sym, bool addMetadata)
{
    OSL_ASSERT (use_optix() && "This function is only supported when using OptiX!");

    std::ostringstream ss;
    ss.imbue (std::locale::classic());  // force C locale
    if (sym.typespec().is_string()) {
        // Use the ustring hash to create a name for the symbol that's based on
        // the string contents
        //
        // TODO: Collisions between variable names are unlikely, but stil
        //       possible. Using something like a counter to handle collisions
        //       is unattractive because it depends on the order in which
        //       strings are encountered at run time.
        //
        //       For now I am simply appending the length to the hash, but a
        //       more robust solution may be called for.

        ss << "ds_"
           << std::setbase (16) << std::setfill('0') << std::setw (16)
           << (*(ustring *)sym.data()).hash()
           << "_"
           << std::setbase (16) << std::setfill('0') << std::setw (4)
           << (*(ustring *)sym.data()).length();

        auto it = m_varname_map.find(ss.str());
        const std::string old_str = (it != m_varname_map.end())
            ? it->second : "";

        if (old_str != "" && old_str != (*(ustring *)sym.data()).string()) {
            std::cerr << "Warning: variable name collision between " << old_str
                      << " and " << (*(ustring *)sym.data()).string()
                      << std::endl;
        }
    }
    else {
        std::string var_name = Strutil::sprintf ("%s_%s_%d_%s_%d",
                                                 sym.name(),
                                                 group().name(),
                                                 group().id(),
                                                 inst()->layername(),
                                                 sym.layer());

        // Leading dollar signs are not allowed in PTX variable names,
        // so prepend an underscore.
        if (var_name[0] == '$') {
            ss << '_';
        }

        ss << var_name;
    }

    const std::string name = ss.str();

    // Return the Value if it has already been allocated
    std::map<std::string, llvm::GlobalVariable*>::iterator it =
        get_const_map().find (name);

    if (it != get_const_map().end()) {
        return it->second;
    }

    // Add the extra metadata needed to make the variable visible to OptiX.
    if (addMetadata || sym.typespec().is_string())
        createOptixMetadata (name, sym);

    // TODO: Figure out the actual CUDA alignment requirements for the various
    //       OSL types. For now, be somewhat conservative and assume 8 for
    //       non-scalar types.
    int alignment = (sym.typespec().is_scalarnum()) ? 4 : 8;

    llvm::Value* cuda_var = addCUDAVariable (name, sym.size(), alignment, sym.data(),
                                             sym.typespec().simpletype());

    return cuda_var;
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
        if (use_optix()) {
            // Check the constant map for the named Symbol; if it's found, then
            // a GlobalVariable has been created for it
            llvm::Value* ptr = getOrAllocateCUDAVariable (sym);
            if (ptr) {
                llvm::Type *cast_type = (! sym.typespec().is_string())
                    ? ll.type_ptr (llvm_type(sym.typespec().elementtype()))
                    : ll.type_void_ptr();

                result = ll.ptr_cast (ptr, cast_type);
            }
        }
        else {
            // For constants, start with *OUR* pointer to the constant values.
            result = ll.ptr_cast (ll.constant_ptr (sym.data()),
                                  ll.type_ptr (llvm_type(sym.typespec().elementtype())));
        }

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
    OSL_DASSERT (sym.typespec().is_array() == (arrayindex != NULL));

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
        OSL_ASSERT (0 && "unhandled constant type");
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
    else if (type.is_string() && cast == TypeDesc::LONGLONG)
        result = ll.ptr_to_cast (result, ll.type_longlong());

    return result;
}



llvm::Value *
BackendLLVM::llvm_load_device_string (const Symbol& sym, bool follow)
{
    // TODO: need to make this work with arrays of strings
    OSL_ASSERT (use_optix() && "This is only intended to be used with CUDA");

    // Recover the userdata index for non-constant parameters
    int userdata_index = find_userdata_index (sym);

    llvm::Value* val = NULL;
    if (sym.symtype() == SymTypeLocal) {
        // Handle temporary local variables
        val = getOrAllocateLLVMSymbol (sym);
        val = ll.ptr_cast (val, ll.type_longlong_ptr());
    }
    else if (userdata_index < 0) {
        // Handle non-varying variables
        OSL_DASSERT (sym.data() && "NULL data in non-varying string");
        val = getOrAllocateCUDAVariable (sym);
    }
    else {
        // Handle potentially varying variables
        val = ll.ptr_cast (groupdata_field_ptr (2 + userdata_index),
                           ll.type_longlong_ptr());
    }

    // It's preferable to deal with device strings "symbolically" through the
    // CUDA variable (essentially a char**), which helps keep the code
    // portable. But sometimes it's necessary to handle the underlying char*
    // directly, e.g. when printing or writing out a closure param.
    if (follow)
        val = ll.int_to_ptr_cast (ll.op_load (val));

    return val;
}



llvm::Value *
BackendLLVM::llvm_load_constant_value (const Symbol& sym, 
                                       int arrayindex, int component,
                                       TypeDesc cast)
{
    OSL_DASSERT (sym.is_constant() &&
                 "Called llvm_load_constant_value for a non-constant symbol");

    // set array indexing to zero for non-arrays
    if (! sym.typespec().is_array())
        arrayindex = 0;
    OSL_DASSERT (arrayindex >= 0 &&
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
    if (sym.typespec().is_string() && use_optix()) {
        OSL_DASSERT ((arrayindex == 0) && "String arrays are not currently supported in OptiX");
        return llvm_load_device_string (sym, /*follow*/ false);
    }
    if (sym.typespec().is_string()) {
        const ustring *val = (const ustring *)sym.data();
        return ll.constant (val[arrayindex]);
    }

    OSL_ASSERT (0 && "unhandled constant type");
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
        OSL_DASSERT (sym.typespec().is_floatbased() &&
                     "can't ask for derivs of an int");
        return ll.constant (0.0f);
    }

    // Start with the initial pointer to the value's memory location
    llvm::Value* result = llvm_get_pointer (sym, deriv);
    if (!result)
        return NULL;  // Error

    OSL_DASSERT (sym.typespec().simpletype().aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*
    result = ll.ptr_cast (result, ll.type_float_ptr());
    result = ll.GEP (result, component);  // get the component

    // Now grab the value
    return ll.op_load (result);
}



llvm::Value *
BackendLLVM::llvm_load_arg (const Symbol& sym, bool derivs)
{
    OSL_DASSERT (sym.typespec().is_floatbased());
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

    OSL_DASSERT (sym.typespec().simpletype().aggregate != TypeDesc::SCALAR);
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
                                 cspan<const Symbol *> args,
                                 bool deriv_ptrs)
{
    // most invocations of this function will only need a handful of args
    // so avoid dynamic allocation where possible
    constexpr int SHORT_NUM_ARGS = 16;
    llvm::Value *short_valargs[SHORT_NUM_ARGS];
    std::vector<llvm::Value*> long_valargs;
    llvm::Value **valargs = short_valargs;
    if (args.size() > SHORT_NUM_ARGS) {
        long_valargs.resize(args.size());
        valargs = long_valargs.data();
    }
    for (int i = 0, nargs = args.size();  i < nargs; ++i) {
        const Symbol &s = *(args[i]);
        if (s.typespec().is_closure())
            valargs[i] = llvm_load_value (s);
        else if (use_optix() && s.typespec().is_string())
            valargs[i] = llvm_load_device_string (s, /*follow*/ true);
        else if (s.typespec().simpletype().aggregate > 1 ||
                 (deriv_ptrs && s.has_derivs()))
            valargs[i] = llvm_void_ptr (s);
        else
            valargs[i] = llvm_load_value (s);
    }
    return ll.call_function (name, cspan<llvm::Value*>(valargs, args.size()));
}

llvm::Value *
BackendLLVM::llvm_test_nonzero (Symbol &val, bool test_derivs)
{
    const TypeSpec &ts (val.typespec());
    OSL_DASSERT (! ts.is_array() && ! ts.is_closure() && ! ts.is_string());
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
                                    int arrayindex, int srccomp, int dstcomp)
{
    OSL_DASSERT (! Result.typespec().is_structure());
    OSL_DASSERT (! Src.typespec().is_structure());

    const TypeSpec &result_t (Result.typespec());
    const TypeSpec &src_t (Src.typespec());

    llvm::Value *arrind = arrayindex >= 0 ? ll.constant (arrayindex) : NULL;

    if (Result.typespec().is_closure() || Src.typespec().is_closure()) {
        if (Src.typespec().is_closure()) {
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
        OSL_DASSERT (assignable(result_t.elementtype(), src_t.elementtype()));
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
    const int num_components = rt.aggregate;
    const bool singlechan = (srccomp != -1) || (dstcomp != -1);
    if (use_optix() && Src.typespec().is_string()) {
        llvm::Value* src = llvm_load_device_string (Src, /*follow*/ true);
        llvm_store_value (ll.ptr_cast (src, ll.type_void_ptr()), Result);
    }
    else if (!singlechan) {
        for (int i = 0; i < num_components; ++i) {
            llvm::Value* src_val = Src.is_constant()
                ? llvm_load_constant_value (Src, arrayindex, i, basetype)
                : llvm_load_value (Src, 0, arrind, i, basetype);
            if (!src_val)
                return false;
            llvm_store_value (src_val, Result, 0, arrind, i);
        }
    } else {
        // connect individual component of an aggregate type
        // set srccomp to 0 for case when src is actually a float
        if (srccomp == -1) srccomp = 0;
        llvm::Value* src_val = Src.is_constant()
            ? llvm_load_constant_value (Src, arrayindex, srccomp, basetype)
            : llvm_load_value (Src, 0, arrind, srccomp, basetype);
        if (!src_val)
            return false;
        // write source float into all compnents when dstcomp == -1, otherwise
        // the single element requested.
        if (dstcomp == -1) {
            for (int i = 0; i < num_components; ++i)
                llvm_store_value (src_val, Result, 0, arrind, i);
        } else
            llvm_store_value (src_val, Result, 0, arrind, dstcomp);
    }

    // Handle derivatives
    if (Result.has_derivs()) {
        if (Src.has_derivs()) {
            // src and result both have derivs -- copy them
            if (!singlechan) {
                for (int d = 1;  d <= 2;  ++d) {
                    for (int i = 0; i < num_components; ++i) {
                        llvm::Value* val = llvm_load_value (Src, d, arrind, i);
                        llvm_store_value (val, Result, d, arrind, i);
                    }
                }
            } else {
                for (int d = 1;  d <= 2;  ++d) {
                    llvm::Value* val = llvm_load_value (Src, d, arrind, srccomp);
                    if (dstcomp == -1) {
                        for (int i = 0; i < num_components; ++i)
                            llvm_store_value (val, Result, d, arrind, i);
                    }
                    else
                        llvm_store_value (val, Result, d, arrind, dstcomp);
                }
            }
        } else {
            // Result wants derivs but src didn't have them -- zero them
            if (dstcomp != -1) {
                // memset the single deriv component's to zero
                if (Result.has_derivs() && Result.typespec().elementtype().is_floatbased()) {
                    // dx
                    ll.op_memset (ll.GEP(llvm_void_ptr(Result,1), dstcomp), 0, 1, rt.basesize());
                    // dy
                    ll.op_memset (ll.GEP(llvm_void_ptr(Result,2), dstcomp), 0, 1, rt.basesize());
                }
            } else
                llvm_zero_derivs (Result);
        }
    }
    return true;
}



int BackendLLVM::find_userdata_index (const Symbol& sym)
{
    int userdata_index = -1;
    for (int i = 0, e = (int)group().m_userdata_names.size(); i < e; ++i) {
        if (sym.name() == group().m_userdata_names[i] &&
            equivalent (sym.typespec().simpletype(), group().m_userdata_types[i])) {
            userdata_index = i;
            break;
        }
    }
    return userdata_index;
}



}; // namespace pvt
OSL_NAMESPACE_EXIT
