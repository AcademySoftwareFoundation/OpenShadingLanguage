// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <type_traits>

#include "batched_backendllvm.h"
#include "oslexec_pvt.h"

#include <llvm/ADT/Twine.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace Strings {

// TODO: What qualifies these to move to strdecls.h?
//       Being used in more than one .cpp?

// Shader global strings
static ustring backfacing("backfacing");
static ustring surfacearea("surfacearea");
static ustring object2common("object2common");
static ustring shader2common("shader2common");
static ustring flipHandedness("flipHandedness");
}  // namespace Strings

namespace pvt {

namespace  // Unnamed
{
// The order of names in this table MUST exactly match the
// BatchedShaderGlobals struct in batched_shaderglobals.h,
// as well as the llvm 'sg' type
// defined in BatchedBackendLLVM::llvm_type_sg().
static ustring fields[] = {
    // Uniform
    ustring("renderstate"),     //
    ustring("tracedata"),       //
    ustring("objdata"),         //
    ustring("shadingcontext"),  //
    ustring("renderer"),        //
    Strings::Ci,                //
    Strings::raytype,           //
    ustring("pad0"),            //
    ustring("pad1"),            //
    ustring("pad2"),            //
    // Varying
    Strings::P,               //
    ustring("dPdz"),          //
    Strings::I,               //
    Strings::N,               //
    Strings::Ng,              //
    Strings::u,               //
    Strings::v,               //
    Strings::dPdu,            //
    Strings::dPdv,            //
    Strings::time,            //
    Strings::dtime,           //
    Strings::dPdtime,         //
    Strings::Ps,              //
    Strings::object2common,   //
    Strings::shader2common,   //
    Strings::surfacearea,     //
    Strings::flipHandedness,  //
    Strings::backfacing
};

static bool field_is_uniform[] = {
    // Uniform
    true,  // renderstate
    true,  // tracedata
    true,  // objdata
    true,  // shadingcontext
    true,  // renderer
    true,  // Ci
    true,  // raytype
    true,  // pad0
    true,  // pad1
    true,  // pad2
    // Varying
    false,  // P
    false,  // dPdz
    false,  // I
    false,  // N
    false,  // Ng
    false,  // u
    false,  // v
    false,  // dPdu
    false,  // dPdv
    false,  // time
    false,  // dtime
    false,  // dPdtime
    false,  // Ps
    false,  // object2common
    false,  // shader2common
    false,  // surfacearea
    false,  // flipHandedness
    false,  // backfacing
};

}  // namespace

extern bool
is_shader_global_uniform_by_name(ustring name)
{
    for (int i = 0; i < int(std::extent<decltype(fields)>::value); ++i) {
        if (name == fields[i]) {
            return field_is_uniform[i];
        }
    }
    return false;
}

BatchedBackendLLVM::BatchedBackendLLVM(ShadingSystemImpl& shadingsys,
                                       ShaderGroup& group, ShadingContext* ctx,
                                       int width)
    : OSOProcessorBase(shadingsys, group, ctx)
    , ll(ctx->llvm_thread_info(), llvm_debug(), width)
    , m_width(width)
    , m_library_selector(nullptr)
    , m_stat_total_llvm_time(0)
    , m_stat_llvm_setup_time(0)
    , m_stat_llvm_irgen_time(0)
    , m_stat_llvm_opt_time(0)
    , m_stat_llvm_jit_time(0)
{
    m_wide_arg_prefix = "W";
    switch (vector_width()) {
    case 16: m_true_mask_value = Mask<16>(true).value(); break;
    case 8: m_true_mask_value = Mask<8>(true).value(); break;
    default: OSL_ASSERT(0 && "unsupported vector width");
    }
}



BatchedBackendLLVM::~BatchedBackendLLVM() {}



int
BatchedBackendLLVM::llvm_debug() const
{
    if (shadingsys().llvm_debug() == 0)
        return 0;
    if (!shadingsys().debug_groupname().empty()
        && shadingsys().debug_groupname() != group().name()) {
        return 0;
    }
    if (inst() && !shadingsys().debug_layername().empty()
        && shadingsys().debug_layername() != inst()->layername())
        return 0;
    return shadingsys().llvm_debug();
}


void
BatchedBackendLLVM::set_inst(int layer)
{
    OSOProcessorBase::set_inst(layer);  // parent does the heavy lifting
    ll.debug(llvm_debug());
}



llvm::Type*
BatchedBackendLLVM::llvm_pass_type(const TypeSpec& typespec)
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
        lt = (llvm::Type*)ll.type_string();
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
    else {
        std::cerr << "Bad llvm_pass_type(" << typespec.c_str() << ")\n";
        OSL_ASSERT(0 && "not handling this type yet");
    }
    if (t.arraylen) {
        OSL_ASSERT(0 && "should never pass an array directly as a parameter");
    }
    return lt;
}

llvm::Type*
BatchedBackendLLVM::llvm_pass_wide_type(const TypeSpec& typespec)
{
    if (typespec.is_closure_based())
        return (llvm::Type*)ll.type_void_ptr();
    TypeDesc t     = typespec.simpletype().elementtype();
    llvm::Type* lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = (llvm::Type*)ll.type_void_ptr();  // ll.type_wide_float();
    else if (t == TypeDesc::INT)
        lt = (llvm::Type*)ll.type_void_ptr();  // ll.type_wide_int();
    else if (t == TypeDesc::STRING)
        lt = (llvm::Type*)
                 ll.type_void_ptr();  // (llvm::Type *) ll.type_wide_ string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = (llvm::Type*)ll.type_void_ptr();  //llvm_type_wide_triple_ptr();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = (llvm::Type*)ll.type_void_ptr();  //llvm_type_wide_matrix_ptr();
    else if (t == TypeDesc::NONE)
        lt = ll.type_void();
    else if (t == TypeDesc::PTR)
        lt = (llvm::Type*)ll.type_void_ptr();
    else if (t == TypeDesc::LONGLONG)
        lt = (llvm::Type*)ll.type_void_ptr();  // ll.type_wide_longlong();
    else {
        std::cerr << "Bad llvm_pass_type(" << typespec.c_str() << ")\n";
        OSL_ASSERT(0 && "not handling this type yet");
    }
    if (t.arraylen) {
        OSL_ASSERT(0 && "should never pass an array directly as a parameter");
    }
    return lt;
}



void
BatchedBackendLLVM::llvm_assign_zero(const Symbol& sym)
{
    llvm::Value* zero;

    const TypeSpec& t = sym.typespec();
    TypeSpec elemtype = t.elementtype();
    if (elemtype.is_float_based()) {
        if (sym.is_uniform())
            zero = ll.constant(0.0f);
        else
            zero = ll.wide_constant(0.0f);
    } else if (elemtype.is_int_based()) {
        if (sym.is_uniform())
            zero = ll.constant(0);
        else
            zero = ll.wide_constant(0);
    } else if (elemtype.is_string_based()) {
        if (sym.is_uniform())
            zero = ll.constant(ustring());
        else
            zero = ll.wide_constant(ustring());
    } else {
        OSL_ASSERT(0 && "Unsupported element type");
        zero = nullptr;
    }

    int num_elements = t.numelements();
    for (int a = 0; a < num_elements; ++a) {
        int numDeriv = sym.has_derivs() ? 3 : 1;
        for (int d = 0; d < numDeriv; ++d) {
            llvm::Value* arrind = t.simpletype().arraylen ? ll.constant(a)
                                                          : NULL;
            for (int c = 0; c < t.aggregate(); ++c) {
                llvm_store_value(zero, sym, d, arrind, c);
            }
        }
    }
}



void
BatchedBackendLLVM::llvm_zero_derivs(const Symbol& sym)
{
    const TypeSpec& t = sym.typespec();

    if (t.is_closure_based())
        return;  // Closures don't have derivs

    TypeSpec elemtype = t.elementtype();
    if (sym.has_derivs() && elemtype.is_float_based()) {
        llvm::Value* zero;
        if (sym.is_uniform())
            zero = ll.constant(0.0f);
        else
            zero = ll.wide_constant(0.0f);

        int start_array_index = -1;
        int end_array_index   = start_array_index + 1;
        if (t.is_array()) {
            // TODO: investigate doing a memset for arrays & matrices,
            // but not for simple aggregates
            start_array_index = 0;
            end_array_index   = t.arraylength();
        }

        for (int arrayindex = start_array_index; arrayindex < end_array_index;
             ++arrayindex) {
            llvm::Value* arrind = arrayindex >= 0 ? ll.constant(arrayindex)
                                                  : NULL;

            for (int c = 0; c < t.aggregate(); ++c)
                llvm_store_value(zero, sym, 1, arrind, c);
            for (int c = 0; c < t.aggregate(); ++c)
                llvm_store_value(zero, sym, 2, arrind, c);
        }
    }
}



void
BatchedBackendLLVM::llvm_zero_derivs(const Symbol& sym, llvm::Value* count)
{
    if (sym.typespec().is_closure_based())
        return;  // Closures don't have derivs
    // Same thing as the above version but with just the first count derivs
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_float_based()) {
        OSL_ASSERT(
            sym.is_uniform());  // TODO: handle varying case and remove OSL_ASSERT
        size_t esize = sym.typespec().simpletype().elementsize();
        size_t align = sym.typespec().simpletype().basesize();
        count        = ll.op_mul(count, ll.constant((int)esize));
        ll.op_memset(llvm_void_ptr(sym, 1), 0, count, (int)align);  // X derivs
        ll.op_memset(llvm_void_ptr(sym, 2), 0, count, (int)align);  // Y derivs
    }
}

int
BatchedBackendLLVM::ShaderGlobalNameToIndex(ustring name, bool& is_uniform)
{
    for (int i = 0; i < int(sizeof(fields) / sizeof(fields[0])); ++i)
        if (name == fields[i]) {
            is_uniform = field_is_uniform[i];
            return i;
        }
    OSL_DEV_ONLY(std::cout << "ShaderGlobalNameToIndex failed with " << name
                           << std::endl);
    return -1;
}

llvm::Value*
BatchedBackendLLVM::llvm_global_symbol_ptr(ustring name, bool& is_uniform)
{
    // Special case for globals -- they live in the ShaderGlobals struct,
    // we use the name of the global to find the index of the field within
    // the ShaderGlobals struct.
    int sg_index = ShaderGlobalNameToIndex(name, is_uniform);
    OSL_ASSERT(sg_index >= 0);
    return ll.void_ptr(ll.GEP(sg_ptr(), 0, sg_index));
}

llvm::Value*
BatchedBackendLLVM::getLLVMSymbolBase(const Symbol& sym)
{
    Symbol* dealiased = sym.dealias();

    bool is_uniform = sym.is_uniform();

    if (sym.symtype() == SymTypeGlobal) {
        llvm::Value* result = llvm_global_symbol_ptr(sym.name(), is_uniform);
        OSL_ASSERT(result);
        if (is_uniform) {
            result = ll.ptr_to_cast(result,
                                    llvm_type(sym.typespec().elementtype()));
        } else {
            result = ll.ptr_to_cast(result, llvm_wide_type(
                                                sym.typespec().elementtype()));
        }
        return result;
    }

    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        // Special case for params -- they live in the group data
        int fieldnum = m_param_order_map[&sym];
        return groupdata_field_ptr(fieldnum,
                                   sym.typespec().elementtype().simpletype(),
                                   is_uniform);
    }

    std::string mangled_name         = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);
    if (map_iter == named_values().end()) {
        shadingcontext()->errorf(
            "Couldn't find symbol '%s' (unmangled = '%s'). Did you forget to allocate it?",
            mangled_name.c_str(), dealiased->name().c_str());
        return 0;
    }
    return (llvm::Value*)map_iter->second;
}

llvm::Value*
BatchedBackendLLVM::llvm_alloca(const TypeSpec& type, bool derivs,
                                bool is_uniform, bool forceBool,
                                const std::string& name)
{
    OSL_DEV_ONLY(std::cout << "llvm_alloca " << name);
    TypeDesc t = llvm_typedesc(type);
    int n      = derivs ? 3 : 1;
    OSL_DEV_ONLY(std::cout << " n=" << n << " t.size()=" << t.size());
    m_llvm_local_mem += t.size() * n;
    if (is_uniform) {
        OSL_DEV_ONLY(std::cout << " as UNIFORM " << std::endl);
        if (forceBool) {
            return ll.op_alloca(ll.type_bool(), n, name);
        } else {
            return ll.op_alloca(t, n, name);
        }
    } else {
        OSL_DEV_ONLY(std::cout << " as VARYING " << std::endl);
        if (forceBool) {
            return ll.op_alloca(ll.type_native_mask(), n, name);
        } else {
            return ll.wide_op_alloca(t, n, name);
        }
    }
}

BatchedBackendLLVM::TempScope::TempScope(BatchedBackendLLVM& backend)
    : m_backend(backend)
{
    m_backend.m_temp_scopes.push_back(this);
}

BatchedBackendLLVM::TempScope::~TempScope()
{
    OSL_ASSERT(!m_backend.m_temp_scopes.empty()
               && m_backend.m_temp_scopes.back() == this);
    OSL_MAYBE_UNUSED int temp_count = static_cast<int>(
        m_backend.m_temp_allocs.size());
    // Any temps we used will no longer be needed,
    // so we can mark them to be reused
    for (int temp_index : m_in_use_indices) {
        OSL_DASSERT(temp_index < temp_count && temp_index >= 0);
        m_backend.m_temp_allocs[temp_index].in_use = false;
    }
    m_backend.m_temp_scopes.pop_back();
}

llvm::Value*
BatchedBackendLLVM::getOrAllocateTemp(const TypeSpec& type, bool derivs,
                                      bool is_uniform, bool forceBool,
                                      const std::string& name)
{
    OSL_ASSERT(
        !m_temp_scopes.empty()
        && "an instance of BatchedBackendLLVM::TempScope must exist higher up on the stack");

    // Check to see if we have a free temp meeting the request
    // using simple reverse linear search
    int temp_count = static_cast<int>(m_temp_allocs.size());
    for (int temp_index = temp_count - 1; temp_index >= 0; --temp_index) {
        TempAlloc& temp_alloc = m_temp_allocs[temp_index];
        if (!temp_alloc.in_use && temp_alloc.derivs == derivs
            && temp_alloc.is_uniform == is_uniform
            && temp_alloc.forceBool == forceBool) {
            // If we are forcing bool, we don't care about the actual type requested
            if (forceBool || temp_alloc.type == type) {
                llvm::Value* cached_alloc = temp_alloc.llvm_value;

                m_temp_scopes.back()->m_in_use_indices.push_back(temp_index);
                temp_alloc.in_use = true;
                return cached_alloc;
            }
        }
    }

    // No free temp matched the request, so allocate one
    // NOTE: the name will be of the 1st user of the temp, it may get reused out of
    // the cache for other purposes than named.  Debatable if the name should be dropped
    // it may hurt more than help
    llvm::Value* allocation = llvm_alloca(type, derivs, is_uniform, forceBool,
                                          name);
    m_temp_allocs.push_back(TempAlloc { true /*in_use*/, derivs, is_uniform,
                                        forceBool, type, allocation });
    m_temp_scopes.back()->m_in_use_indices.push_back(temp_count);
    return allocation;
}

llvm::Value*
BatchedBackendLLVM::getOrAllocateLLVMSymbol(const Symbol& sym)
{
    OSL_DASSERT(
        (sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp
         || sym.symtype() == SymTypeConst)
        && "getOrAllocateLLVMSymbol should only be for local, tmp, const");
    Symbol* dealiased                = sym.dealias();
    std::string mangled_name         = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);

    if (map_iter == named_values().end()) {
        bool is_uniform = sym.is_uniform();
        bool forceBool  = sym.forced_llvm_bool();

        llvm::Value* a = llvm_alloca(sym.typespec(), sym.has_derivs(),
                                     is_uniform, forceBool, mangled_name);
        named_values()[mangled_name] = a;
        return a;
    }
    return map_iter->second;
}



llvm::Value*
BatchedBackendLLVM::llvm_get_pointer(const Symbol& sym, int deriv,
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
        // For constants, start with *OUR* pointer to the constant values.
        result
            = ll.ptr_cast(ll.constant_ptr(sym.data()),
                          // Constants by definition should always be UNIFORM
                          ll.type_ptr(llvm_type(sym.typespec().elementtype())));

    } else {
        // Start with the initial pointer to the variable's memory location
        result = getLLVMSymbolBase(sym);
#ifdef OSL_DEV
        std::cerr << " llvm_get_pointer(" << sym.name() << ") result=";
        {
            llvm::raw_os_ostream os_cerr(std::cerr);
            ll.llvm_typeof(result)->print(os_cerr);
        }
        std::cerr << std::endl;
#endif
    }
    if (!result)
        return NULL;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t = sym.typespec().simpletype();
    if (t.arraylen || has_derivs) {
#ifdef OSL_DEV
        std::cout << "llvm_get_pointer we're dealing with an array("
                  << t.arraylen << ") or has_derivs(" << has_derivs
                  << ")<<-------" << std::endl;
        std::cout << "arrayindex=" << arrayindex << " deriv=" << deriv
                  << " t.arraylen=" << t.arraylen;
        std::cout << " is_uniform=" << sym.is_uniform() << std::endl;
#endif

        int d = deriv * std::max(1, t.arraylen);
        if (arrayindex)
            arrayindex = ll.op_add(arrayindex, ll.constant(d));
        else
            arrayindex = ll.constant(d);
        result = ll.GEP(result, arrayindex);
    }

    return result;
}

llvm::Value*
BatchedBackendLLVM::llvm_widen_value_into_temp(const Symbol& sym, int deriv)
{
    OSL_ASSERT(
        !m_temp_scopes.empty()
        && "An instance of BatchedBackendLLVM::TempScope must exist higher up in the call stack");
    OSL_ASSERT(sym.is_uniform() == true);
    const TypeSpec& t = sym.typespec();

    TypeDesc symType = t.simpletype();
    OSL_ASSERT(symType.is_unknown() == false);

    llvm::Value* widePtr       = getOrAllocateTemp(t, false /*derivs*/,
                                             false /*is_uniform*/);
    auto disable_masked_stores = ll.create_masking_scope(false);
    for (int c = 0; c < t.aggregate(); ++c) {
        // NOTE: we use the passed deriv to load, but store to value (deriv==0)
        llvm::Value* v = llvm_load_value(sym, deriv, c, TypeDesc::UNKNOWN,
                                         /*is_uniform*/ false);
        llvm_store_value(v, widePtr, t, 0, NULL, c);
    }
    return ll.void_ptr(widePtr);
}

llvm::Value*
BatchedBackendLLVM::llvm_load_value(const Symbol& sym, int deriv,
                                    llvm::Value* arrayindex, int component,
                                    TypeDesc cast, bool op_is_uniform,
                                    bool index_is_uniform)
{
    // A uniform symbol can be broadcast into a varying value.
    // But a varying symbol can NOT be loaded into a uniform value.
    OSL_ASSERT(!op_is_uniform || sym.is_uniform());
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        if (op_is_uniform) {
            return ll.constant(0.0f);
        } else {
            return ll.wide_constant(0.0f);
        }
    }

    // arrayindex should be non-NULL if and only if sym is an array
    OSL_ASSERT(sym.typespec().is_array() == (arrayindex != NULL));

    if (sym.is_constant() && !sym.typespec().is_array() && !arrayindex) {
        // Shortcut for simple constants
        if (sym.typespec().is_float()) {
            if (cast == TypeDesc::TypeInt)
                if (op_is_uniform) {
                    return ll.constant((int)*(float*)sym.data());
                } else {
                    return ll.wide_constant((int)*(float*)sym.data());
                }
            else if (op_is_uniform) {
                return ll.constant(*(float*)sym.data());
            } else {
                return ll.wide_constant(*(float*)sym.data());
            }
        }
        if (sym.typespec().is_int()) {
            if (cast == TypeDesc::TypeFloat)
                if (op_is_uniform) {
                    return ll.constant((float)*(int*)sym.data());
                } else {
                    return ll.wide_constant((float)*(int*)sym.data());
                }
            else {
                if (op_is_uniform) {
                    return ll.constant(*(int*)sym.data());
                } else {
                    return ll.wide_constant(*(int*)sym.data());
                }
            }
        }
        if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
            if (op_is_uniform) {
                return ll.constant(((float*)sym.data())[component]);
            } else {
                return ll.wide_constant(((float*)sym.data())[component]);
            }
        }
        if (sym.typespec().is_string()) {
            if (op_is_uniform) {
                return ll.constant(*(ustring*)sym.data());
            } else {
                return ll.wide_constant(*(ustring*)sym.data());
            }
        }
        OSL_ASSERT(0 && "unhandled constant type");
    }

    OSL_DEV_ONLY(std::cout << "  llvm_load_value " << sym.typespec().string()
                           << " cast " << cast << std::endl);
    return llvm_load_value(llvm_get_pointer(sym), sym.typespec(), deriv,
                           arrayindex, component, cast, op_is_uniform,
                           index_is_uniform, sym.forced_llvm_bool());
}


llvm::Value*
BatchedBackendLLVM::llvm_load_mask(const Symbol& cond)
{
    OSL_ASSERT(cond.is_varying());
    OSL_ASSERT(cond.typespec().is_int());
    llvm::Value* llvm_mask = nullptr;
    llvm::Value* llvm_mask_or_wide_int
        = llvm_load_value(cond, /*deriv*/ 0, /*component*/ 0,
                          /*cast*/ TypeDesc::UNKNOWN, /*op_is_uniform*/ false);
    if (cond.forced_llvm_bool()) {
        // The llvm_load_value + TypeDesc::UNKNOWN will check and convert to llvm mask already
        llvm_mask = llvm_mask_or_wide_int;
    } else {
        OSL_ASSERT(ll.llvm_typeof(llvm_mask_or_wide_int) == ll.type_wide_int());
        llvm_mask = ll.op_int_to_bool(llvm_mask_or_wide_int);
    }

    OSL_ASSERT(ll.llvm_typeof(llvm_mask) == ll.type_wide_bool());
    return llvm_mask;
}


llvm::Value*
BatchedBackendLLVM::llvm_load_value(llvm::Value* ptr, const TypeSpec& type,
                                    int deriv, llvm::Value* arrayindex,
                                    int component, TypeDesc cast,
                                    bool op_is_uniform, bool index_is_uniform,
                                    bool symbol_forced_boolean)
{
    if (!ptr)
        return NULL;  // Error

    if (index_is_uniform) {
        // If it's an array or we're dealing with derivatives, step to the
        // right element.
        TypeDesc t = type.simpletype();
        if (t.arraylen || deriv) {
            int d = deriv * std::max(1, t.arraylen);
            llvm::Value* elem;
            if (arrayindex)
                elem = ll.op_add(arrayindex, ll.constant(d));
            else
                elem = ll.constant(d);
            ptr = ll.GEP(ptr, elem);
        }

        // If it's multi-component (triple or matrix), step to the right field
        if (!type.is_closure_based() && t.aggregate > 1) {
            OSL_DEV_ONLY(std::cout << "step to the right field " << component
                                   << std::endl);
            ptr = ll.GEP(ptr, 0, component);
        }

        // Now grab the value
        llvm::Value* result;
        result = ll.op_load(ptr);

        if (type.is_closure_based())
            return result;

        // We may have bool masquarading as int's and need to promote them for
        // use in any int arithmetic
        if (type.is_int() && symbol_forced_boolean) {
            // We only need to convert wide native masks
            // and op_is_uniform doesn't guarantee that the symbol it self
            // in non-unform, it could just be a single bool (vs. mask).
            if (!op_is_uniform && (ll.llvm_typeof(result) != ll.type_bool())) {
                // We just loaded a native mask need to convert it
                // to a vector of bools
                result = ll.native_to_llvm_mask(result);
            }

            if (cast != TypeDesc::UNKNOWN) {
                if (cast == TypeDesc::TypeInt) {
                    result = ll.op_bool_to_int(result);
                } else if (cast == TypeDesc::TypeFloat) {
                    result = ll.op_bool_to_float(result);
                }
            }
        }

        // Handle int<->float type casting
        if (type.is_float_based() && cast == TypeDesc::TypeInt)
            result = ll.op_float_to_int(result);
        else if (type.is_int() && cast == TypeDesc::TypeFloat)
            result = ll.op_int_to_float(result);
        else if (type.is_string() && cast == TypeDesc::LONGLONG)
            result = ll.ptr_to_cast(result, ll.type_longlong());

        if (!op_is_uniform) {
            // TODO:  remove this assert once we have confirmed correct handling off all the
            // different data types.  Using OSL_ASSERT as a checklist to verify what we have
            // handled so far during development
            OSL_ASSERT(
                cast == TypeDesc::UNKNOWN || cast == TypeDesc::TypeColor
                || cast == TypeDesc::TypeVector || cast == TypeDesc::TypePoint
                || cast == TypeDesc::TypeNormal || cast == TypeDesc::TypeFloat
                || cast == TypeDesc::TypeInt || cast == TypeDesc::TypeString
                || cast == TypeDesc::TypeMatrix || cast == TypeDesc::LONGLONG);

            if ((ll.llvm_typeof(result) == ll.type_bool())
                || (ll.llvm_typeof(result) == ll.type_float())
                || (ll.llvm_typeof(result) == ll.type_triple())
                || (ll.llvm_typeof(result) == ll.type_int())
                || (ll.llvm_typeof(result) == (llvm::Type*)ll.type_string())
                || (ll.llvm_typeof(result) == ll.type_matrix())
                || (ll.llvm_typeof(result) == ll.type_longlong())) {
                result = ll.widen_value(result);
            } else {
#ifdef OSL_DEV
                if (!((ll.llvm_typeof(result) == ll.type_wide_float())
                      || (ll.llvm_typeof(result) == ll.type_wide_int())
                      || (ll.llvm_typeof(result) == ll.type_wide_matrix())
                      || (ll.llvm_typeof(result) == ll.type_wide_triple())
                      || (ll.llvm_typeof(result) == ll.type_wide_string())
                      || (ll.llvm_typeof(result) == ll.type_wide_bool()))) {
                    OSL_DEV_ONLY(std::cout << ">>>>>>>>>>>>>> TYPENAME OF "
                                           << ll.llvm_typenameof(result)
                                           << std::endl);
                }
#endif
                OSL_ASSERT(
                    (ll.llvm_typeof(result) == ll.type_wide_float())
                    || (ll.llvm_typeof(result) == ll.type_wide_int())
                    || (ll.llvm_typeof(result) == ll.type_wide_triple())
                    || (ll.llvm_typeof(result) == ll.type_wide_string())
                    || (ll.llvm_typeof(result) == ll.type_wide_bool())
                    || (ll.llvm_typeof(result) == ll.type_wide_matrix())
                    || (ll.llvm_typeof(result) == ll.type_wide_longlong()));
            }
        }
        return result;
    } else {
        OSL_ASSERT(!op_is_uniform);
        OSL_ASSERT(nullptr != arrayindex);
        // If it's an array or we're dealing with derivatives, step to the
        // right element.
        TypeDesc t = type.simpletype();
        if (t.arraylen || deriv) {
            int d             = deriv * std::max(1, t.arraylen);
            llvm::Value* elem = ll.constant(d);
            ptr               = ll.GEP(ptr, elem);
        }

        // If it's multi-component (triple or matrix), step to the right field
        if (!type.is_closure_based() && t.aggregate > 1) {
            OSL_DEV_ONLY(std::cout << "step to the right field " << component
                                   << std::endl);
            ptr = ll.GEP(ptr, 0, component);

            // Need to scale the indices by the stride
            // of the type
            int elem_stride = t.aggregate;
            arrayindex = ll.op_mul(arrayindex, ll.wide_constant(elem_stride));
            // TODO: possible optimization when elem_stride == 2 && sizeof(type) == 4,
            // could have optional parameter gather operation to use a scale of 8 (2*4)
            // vs. the hardcoded 4 and avoid the multiplication above
        }


        // Now grab the value
        llvm::Value* result;
        result = ll.op_gather(ptr, arrayindex);
        // TODO:  possible optimization when we know the array size is small (<= 4)
        // instead of performing a gather, we could load each value of the the array,
        // compare the index array against that value's index and select/blend
        // the results together.  Basically we will loading the entire content of the
        // array, but can avoid branching or any gather statements.

        if (type.is_closure_based())
            return result;

        OSL_ASSERT(ll.llvm_typeof(result) != ll.type_wide_bool());

        // Handle int<->float type casting
        if (type.is_float_based() && cast == TypeDesc::TypeInt)
            result = ll.op_float_to_int(result);
        else if (type.is_int() && cast == TypeDesc::TypeFloat)
            result = ll.op_int_to_float(result);
        else if (type.is_string() && cast == TypeDesc::LONGLONG)
            result = ll.ptr_to_cast(result, ll.type_longlong());


        return result;
    }
}



llvm::Value*
BatchedBackendLLVM::llvm_load_constant_value(const Symbol& sym, int arrayindex,
                                             int component, TypeDesc cast,
                                             bool op_is_uniform)
{
    OSL_ASSERT(sym.is_constant()
               && "Called llvm_load_constant_value for a non-constant symbol");

    // set array indexing to zero for non-arrays
    if (!sym.typespec().is_array())
        arrayindex = 0;
    OSL_ASSERT(arrayindex >= 0
               && "Called llvm_load_constant_value with negative array index");


    // TODO: might want to take this fix for array types back to the non-wide backend
    TypeSpec elementType = sym.typespec();
    // The symbol we are creating a constant for might be an array
    // and our checks for types use non-array types
    elementType.make_array(0);

    if (elementType.is_float()) {
        const float* val = (const float*)sym.data();
        if (cast == TypeDesc::TypeInt)
            if (op_is_uniform) {
                return ll.constant((int)val[arrayindex]);
            } else {
                return ll.wide_constant((int)val[arrayindex]);
            }
        else if (op_is_uniform) {
            return ll.constant(val[arrayindex]);
        } else {
            return ll.wide_constant(val[arrayindex]);
        }
    }
    if (elementType.is_int()) {
        const int* val = (const int*)sym.data();
        if (cast == TypeDesc::TypeFloat)
            if (op_is_uniform) {
                return ll.constant((float)val[arrayindex]);
            } else {
                return ll.wide_constant((float)val[arrayindex]);
            }
        else if (op_is_uniform) {
            return ll.constant(val[arrayindex]);
        } else {
            return ll.wide_constant(val[arrayindex]);
        }
    }
    if (elementType.is_triple() || elementType.is_matrix()) {
        const float* val = (const float*)sym.data();
        int ncomps       = (int)sym.typespec().aggregate();
        if (op_is_uniform) {
            return ll.constant(val[ncomps * arrayindex + component]);
        } else {
            return ll.wide_constant(val[ncomps * arrayindex + component]);
        }
    }
    if (elementType.is_string()) {
        const ustring* val = (const ustring*)sym.data();
        if (op_is_uniform) {
            return ll.constant(val[arrayindex]);
        } else {
            return ll.wide_constant(val[arrayindex]);
        }
    }

    std::cout << "SYMBOL " << sym.name().c_str() << " type=" << sym.typespec()
              << std::endl;
    OSL_ASSERT(0 && "unhandled constant type");
    return NULL;
}



llvm::Value*
BatchedBackendLLVM::llvm_load_component_value(const Symbol& sym, int deriv,
                                              llvm::Value* component,
                                              bool op_is_uniform,
                                              bool component_is_uniform)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        OSL_ASSERT(sym.typespec().is_float_based()
                   && "can't ask for derivs of an int");
        if (op_is_uniform) {
            return ll.constant(0.0f);
        } else {
            return ll.wide_constant(0.0f);
        }
    }

    // Start with the initial pointer to the value's memory location
    llvm::Value* pointer = llvm_get_pointer(sym, deriv);
    if (!pointer)
        return NULL;  // Error

    TypeDesc t = sym.typespec().simpletype();
    OSL_ASSERT(t.basetype == TypeDesc::FLOAT);
    OSL_ASSERT(t.aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*

    if (sym.is_uniform()) {
        pointer = ll.ptr_cast(pointer, ll.type_float_ptr());
    } else {
        pointer = ll.ptr_cast(pointer, ll.type_wide_float_ptr());
    }

    llvm::Value* result;
    if (component_is_uniform) {
        llvm::Value* component_pointer = ll.GEP(pointer, component);

        // Now grab the value
        result = ll.op_load(component_pointer);
    } else {
        OSL_ASSERT(!op_is_uniform);
        result = ll.op_gather(pointer, component);
        // TODO:  possible optimization when we know the # of components is small (<= 4)
        // instead of performing a gather, we could load each value of the components,
        // compare the component index against that value's index and select/blend
        // the results together.  Basically we will loading the entire content of the
        // object, but can avoid branching or any gather statements.
    }

    if (!op_is_uniform) {
        if (ll.llvm_typeof(result) == ll.type_float()) {
            result = ll.widen_value(result);
        } else {
            OSL_ASSERT(ll.llvm_typeof(result) == ll.type_wide_float());
        }
    }
    return result;
}



llvm::Value*
BatchedBackendLLVM::llvm_load_arg(const Symbol& sym, bool derivs,
                                  bool op_is_uniform)
{
    OSL_ASSERT(
        !m_temp_scopes.empty()
        && "An instance of BatchedBackendLLVM::TempScope must exist higher up in the call stack");
    bool sym_is_uniform = sym.is_uniform();

    if (sym.typespec().is_string() || sym.typespec().is_int()
        || (sym.typespec().is_float() && !derivs)) {
        // Scalar case

        // If we are not uniform, then the argument should
        // get passed as a pointer instead of by value
        // So let this case fall through
        if (op_is_uniform) {
            return llvm_load_value(sym, 0, 0, TypeDesc::UNKNOWN, op_is_uniform);
        } else if (sym.symtype() == SymTypeConst) {
            // As the case to deliver a pointer to a symbol data
            // doesn't provide an opportunity to promote a uniform constant
            // to a wide value that the non-uniform function is expecting
            // we will handle it here.
            llvm::Value* wide_constant_value
                = llvm_load_constant_value(sym, 0, 0, TypeDesc::UNKNOWN,
                                           op_is_uniform);

            // Have to have a place on the stack for the pointer to the wide constant to point to
            const TypeSpec& t   = sym.typespec();
            llvm::Value* tmpptr = getOrAllocateTemp(t, false /*derivs*/,
                                                    false /*is_uniform*/);

            // Store our wide pointer on the stack
            auto disable_masked_stores = ll.create_masking_scope(false);
            llvm_store_value(wide_constant_value, tmpptr, t, 0, NULL, 0);

            // return pointer to our stacked wide constant
            return ll.void_ptr(tmpptr);
        }
    }

    if ((sym_is_uniform && !op_is_uniform) || (derivs && !sym.has_derivs())) {
        // Manufacture-derivs case
        const TypeSpec& t = sym.typespec();

        // Copy the non-deriv values component by component
        llvm::Value* tmpptr = getOrAllocateTemp(t, derivs, op_is_uniform);

        auto disable_masked_stores = ll.create_masking_scope(false);
        int copy_deriv_count       = (derivs && sym.has_derivs()) ? 3 : 1;
        for (int d = 0; d < copy_deriv_count; ++d) {
            for (int c = 0; c < t.aggregate(); ++c) {
                // Will automatically widen value if needed
                llvm::Value* v = llvm_load_value(sym, d, c, TypeDesc::UNKNOWN,
                                                 op_is_uniform);
                llvm_store_value(v, tmpptr, t, d, NULL, c);
            }
        }
        if (derivs && !sym.has_derivs()) {
            // Zero out the deriv values
            llvm::Value* zero;
            if (op_is_uniform)
                zero = ll.constant(0.0f);
            else
                zero = ll.wide_constant(0.0f);
            for (int c = 0; c < t.aggregate(); ++c)
                llvm_store_value(zero, tmpptr, t, 1, NULL, c);
            for (int c = 0; c < t.aggregate(); ++c)
                llvm_store_value(zero, tmpptr, t, 2, NULL, c);
        }
        return ll.void_ptr(tmpptr);
    }

    // Regular pointer case
    return llvm_void_ptr(sym);
}



bool
BatchedBackendLLVM::llvm_store_value(llvm::Value* new_val, const Symbol& sym,
                                     int deriv, llvm::Value* arrayindex,
                                     int component, bool index_is_uniform)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    return llvm_store_value(new_val, llvm_get_pointer(sym), sym.typespec(),
                            deriv, arrayindex, component, index_is_uniform);
}



bool
BatchedBackendLLVM::llvm_store_value(llvm::Value* new_val, llvm::Value* dst_ptr,
                                     const TypeSpec& type, int deriv,
                                     llvm::Value* arrayindex, int component,
                                     bool index_is_uniform)
{
    if (!dst_ptr)
        return false;  // Error

    if (index_is_uniform) {
        // If it's an array or we're dealing with derivatives, step to the
        // right element.
        TypeDesc t = type.simpletype();
        if (t.arraylen || deriv) {
            int d = deriv * std::max(1, t.arraylen);
            if (arrayindex)
                arrayindex = ll.op_add(arrayindex, ll.constant(d));
            else
                arrayindex = ll.constant(d);
            dst_ptr = ll.GEP(dst_ptr, arrayindex);
        }

        // If it's multi-component (triple or matrix), step to the right field
        if (!type.is_closure_based() && t.aggregate > 1)
            dst_ptr = ll.GEP(dst_ptr, 0, component);

        if (ll.type_ptr(ll.llvm_typeof(new_val)) != ll.llvm_typeof(dst_ptr)) {
            std::cerr << " new_val type=";
            {
                llvm::raw_os_ostream os_cerr(std::cerr);
                ll.llvm_typeof(new_val)->print(os_cerr);
            }
            std::cerr << " dest_ptr type=";
            {
                llvm::raw_os_ostream os_cerr(std::cerr);
                ll.llvm_typeof(dst_ptr)->print(os_cerr);
            }
            std::cerr << std::endl;
        }
        OSL_ASSERT(ll.type_ptr(ll.llvm_typeof(new_val))
                   == ll.llvm_typeof(dst_ptr));


        // Finally, store the value.
        ll.op_store(new_val, dst_ptr);
        return true;
    } else {
        OSL_ASSERT(nullptr != arrayindex);

        // If it's an array or we're dealing with derivatives, step to the
        // right element.
        TypeDesc t = type.simpletype();
        if (t.arraylen || deriv) {
            int d             = deriv * std::max(1, t.arraylen);
            llvm::Value* elem = ll.constant(d);
            dst_ptr           = ll.GEP(dst_ptr, elem);
        }

        // If it's multi-component (triple or matrix), step to the right field
        if (!type.is_closure_based() && t.aggregate > 1) {
            OSL_DEV_ONLY(std::cout << "step to the right field " << component
                                   << std::endl);
            dst_ptr = ll.GEP(dst_ptr, 0, component);

            // Need to scale the indices by the stride
            // of the type
            int elem_stride = t.aggregate;
            arrayindex = ll.op_mul(arrayindex, ll.wide_constant(elem_stride));
            // TODO: possible optimization when elem_stride == 2 && sizeof(type) == 4,
            // could have optional parameter gather operation to use a scale of 8 (2*4)
            // vs. the hardcoded 4 and avoid the multiplication above
        }

        // Finally, store the value.
        ll.op_scatter(new_val, dst_ptr, arrayindex);
        // TODO:  possible optimization when we know the array size is small (<= 4)
        // instead of performing a scatter, we could load each value of the the array,
        // compare the index array against that value's index and select/blend
        // the results together, and store the result.  Basically we will loading the entire content of the
        // array, but can avoid branching or any scatter statements.
        return true;
    }
}


bool
BatchedBackendLLVM::llvm_store_mask(llvm::Value* new_mask, const Symbol& cond)
{
    OSL_ASSERT(ll.llvm_typeof(new_mask) == ll.type_wide_bool());
    OSL_ASSERT(cond.is_varying());
    OSL_ASSERT(cond.typespec().is_int());
    if (cond.forced_llvm_bool()) {
        return llvm_store_value(ll.llvm_mask_to_native(new_mask), cond);
    } else {
        return llvm_store_value(ll.op_bool_to_int(new_mask), cond);
    }
}


bool
BatchedBackendLLVM::llvm_store_component_value(llvm::Value* new_val,
                                               const Symbol& sym, int deriv,
                                               llvm::Value* component,
                                               bool component_is_uniform)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    // Let llvm_get_pointer do most of the heavy lifting to get us a
    // pointer to where our data lives.
    llvm::Value* pointer = llvm_get_pointer(sym, deriv);
    if (!pointer)
        return false;  // Error

    TypeDesc t = sym.typespec().simpletype();
    OSL_ASSERT(t.basetype == TypeDesc::FLOAT);
    OSL_ASSERT(t.aggregate != TypeDesc::SCALAR);
    // cast the Vec* to a float*

    bool symbolsIsUniform = sym.is_uniform();
    if (symbolsIsUniform) {
        pointer = ll.ptr_cast(pointer, ll.type_float_ptr());
    } else {
        pointer = ll.ptr_cast(pointer, ll.type_wide_float_ptr());
    }

    if (component_is_uniform) {
        llvm::Value* component_pointer
            = ll.GEP(pointer, component);  // get the component

        // Finally, store the value.
        ll.op_store(new_val, component_pointer);
    } else {
        OSL_ASSERT(!symbolsIsUniform);
        ll.op_scatter(new_val, pointer, component);
    }
    return true;
}

void
BatchedBackendLLVM::llvm_broadcast_uniform_value_from_mem(
    llvm::Value* pointerTotempUniform, const Symbol& Destination,
    bool ignore_derivs)
{
    OSL_ASSERT(Destination.is_varying());
    const TypeDesc& dest_type = Destination.typespec().simpletype();
    bool derivs               = Destination.has_derivs();

    int derivCount = (!ignore_derivs && derivs) ? 3 : 1;

    int arrayEnd;

    if (dest_type.is_array()) {
        OSL_ASSERT(dest_type.arraylen != 0);
        OSL_ASSERT(dest_type.arraylen != -1
                   && "We don't support an unsized array");
        arrayEnd = dest_type.arraylen;
    } else {
        arrayEnd = 1;
    }

    int componentCount = dest_type.aggregate;

    for (int derivIndex = 0; derivIndex < derivCount; ++derivIndex) {
        for (int arrayIndex = 0; arrayIndex < arrayEnd; ++arrayIndex) {
            llvm::Value* llvm_array_index = ll.constant(arrayIndex);
            for (int componentIndex = 0; componentIndex < componentCount;
                 ++componentIndex) {
                // Load the uniform component from the temporary
                // base passing false for op_is_uniform, the llvm_load_value will
                // automatically broadcast the uniform value to a vector type
                llvm::Value* wide_component_value
                    = llvm_load_value(pointerTotempUniform, dest_type,
                                      derivIndex, llvm_array_index,
                                      componentIndex, TypeDesc::UNKNOWN,
                                      false /*op_is_uniform*/);
                bool success = llvm_store_value(wide_component_value,
                                                Destination, derivIndex,
                                                llvm_array_index,
                                                componentIndex);
                OSL_ASSERT(success);
            }
        }
    }
}

void
BatchedBackendLLVM::llvm_broadcast_uniform_value(llvm::Value* tempUniform,
                                                 const Symbol& Destination,
                                                 int derivs, int component)
{
    const TypeDesc& dest_type = Destination.typespec().simpletype();
    OSL_ASSERT(false == dest_type.is_array());

    llvm::Value* wide_value = ll.widen_value(tempUniform);
    llvm_store_value(wide_value, Destination, derivs, nullptr, component);
}

void
BatchedBackendLLVM::llvm_conversion_store_masked_status(llvm::Value* val,
                                                        const Symbol& Status)
{
    OSL_ASSERT(ll.type_int() == ll.llvm_typeof(val));

    llvm::Value* mask = ll.int_as_mask(val);

    if (Status.forced_llvm_bool()) {
        // status is a "native" boolean, so we need to convert it before storing
        mask = ll.llvm_mask_to_native(mask);
    } else {
        llvm::Type* statusType = ll.llvm_typeof(llvm_get_pointer(Status));
        OSL_ASSERT(statusType
                   == reinterpret_cast<llvm::Type*>(ll.type_wide_int_ptr()));
        // status is a integer, so we need to convert it to integer before storing
        mask = ll.op_bool_to_int(mask);
    }

    llvm_store_value(mask, Status);
}

void
BatchedBackendLLVM::llvm_conversion_store_uniform_status(llvm::Value* val,
                                                         const Symbol& Status)
{
    OSL_ASSERT(ll.type_int() == ll.llvm_typeof(val));

    bool is_uniform = Status.is_uniform();
    if (Status.forced_llvm_bool()) {
        // Handle demoting to bool
        val = ll.op_int_to_bool(val);
        if (!is_uniform) {
            // expanding out to wide bool
            val = ll.widen_value(val);
            // Always store native mask type
            val = ll.llvm_mask_to_native(val);
        }
    } else {
        if (!is_uniform) {
            // Expanding out to wide int
            val = ll.widen_value(val);
        }
    }

    llvm_store_value(val, Status);
}

llvm::Value*
BatchedBackendLLVM::groupdata_field_ref(int fieldnum)
{
    return ll.GEP(groupdata_ptr(), 0, fieldnum);
}


llvm::Value*
BatchedBackendLLVM::groupdata_field_ptr(int fieldnum, TypeDesc type,
                                        bool is_uniform)
{
    llvm::Value* result = ll.void_ptr(groupdata_field_ref(fieldnum));
    if (type != TypeDesc::UNKNOWN) {
        if (is_uniform) {
            result = ll.ptr_to_cast(result, llvm_type(type));
        } else {
            result = ll.ptr_to_cast(result, llvm_wide_type(type));
        }
    }
    return result;
}

llvm::Value*
BatchedBackendLLVM::temp_wide_matrix_ptr()
{
    if (m_llvm_temp_wide_matrix_ptr == nullptr) {
        // Don't worry about what basic block we are currently inside of because
        // we insert all alloca's to the top function, not the current insertion point
        m_llvm_temp_wide_matrix_ptr = ll.op_alloca(ll.type_wide_matrix(), 1,
                                                   std::string(), 64);
    }
    return m_llvm_temp_wide_matrix_ptr;
}


llvm::Value*
BatchedBackendLLVM::temp_batched_texture_options_ptr()
{
    if (m_llvm_temp_batched_texture_options_ptr == nullptr) {
        // Don't worry about what basic block we are currently inside of because
        // we insert all alloca's to the top function, not the current insertion point
        m_llvm_temp_batched_texture_options_ptr
            = ll.op_alloca(llvm_type_batched_texture_options(), 1,
                           std::string(), 64);
    }
    return m_llvm_temp_batched_texture_options_ptr;
}

llvm::Value*
BatchedBackendLLVM::temp_batched_trace_options_ptr()
{
    if (m_llvm_temp_batched_trace_options_ptr == nullptr) {
        // Don't worry about what basic block we are currently inside of because
        // we insert all alloca's to the top function, not the current insertion point
        m_llvm_temp_batched_trace_options_ptr
            = ll.op_alloca(llvm_type_batched_trace_options(), 1, std::string(),
                           64);
    }
    return m_llvm_temp_batched_trace_options_ptr;
}


llvm::Value*
BatchedBackendLLVM::layer_run_ref(int layer)
{
    int fieldnum           = 0;  // field 0 is the layer_run array
    llvm::Value* layer_run = groupdata_field_ref(fieldnum);
    return ll.GEP(layer_run, 0, layer);
}



llvm::Value*
BatchedBackendLLVM::userdata_initialized_ref(int userdata_index)
{
    int fieldnum = 1;  // field 1 is the userdata_initialized array
    llvm::Value* userdata_initiazlied = groupdata_field_ref(fieldnum);
    return ll.GEP(userdata_initiazlied, 0, userdata_index);
}


llvm::Value*
BatchedBackendLLVM::llvm_call_function(const FuncSpec& name,
                                       const Symbol** symargs, int nargs,
                                       bool deriv_ptrs,
                                       bool function_is_uniform,
                                       bool functionIsLlvmInlined,
                                       bool ptrToReturnStructIs1stArg)
{
    bool requiresMasking = ptrToReturnStructIs1stArg && !function_is_uniform
                           && ll.is_masking_required();

    llvm::Value* uniformResultTemp = nullptr;

    TempScope scoped_temps(*this);

    bool needToBroadcastUniformResultToWide = false;
    std::vector<llvm::Value*> valargs;
    valargs.resize((size_t)nargs + (requiresMasking ? 1 : 0));
    for (int i = 0; i < nargs; ++i) {
        const Symbol& s   = *(symargs[i]);
        const TypeSpec& t = s.typespec();

        if (t.is_closure())
            valargs[i] = llvm_load_value(s);
        else if (t.simpletype().aggregate > 1 || (deriv_ptrs && s.has_derivs())
                 || (!function_is_uniform && !functionIsLlvmInlined)) {
            // Need to pass a pointer to the function
            if (function_is_uniform || (s.symtype() != SymTypeConst)) {
                //OSL_ASSERT(function_is_uniform || s.is_varying());
                if (function_is_uniform) {
                    OSL_DASSERT(function_is_uniform);
                    if (s.is_uniform()) {
                        valargs[i] = llvm_void_ptr(s);
                    } else {
                        // if the function is uniform, all parameters need to be uniform
                        // however we could doing a masked assignment to a wide result
                        // which would have to be the 1st parameter!
                        OSL_ASSERT(i == 0);
                        OSL_ASSERT(ptrToReturnStructIs1stArg);
                        // in that case we will allocate a uniform result parameter on the
                        // stack to hold the result and then do a masked store after
                        // calling the uniform version of the function
                        OSL_ASSERT(!t.is_array() && "incomplete");
                        needToBroadcastUniformResultToWide = true;
                        uniformResultTemp
                            = getOrAllocateTemp(t, s.has_derivs(),
                                                /*is_uniform=*/true);

                        valargs[i] = ll.void_ptr(uniformResultTemp);
                    }
                } else if (s.is_varying()) {
                    OSL_DASSERT(false == function_is_uniform);
                    valargs[i] = llvm_void_ptr(s);
                } else {
                    OSL_DASSERT(false == function_is_uniform);
                    OSL_DASSERT(s.is_uniform());
                    // TODO: Consider dynamically generating function name based on varying/uniform parameters
                    // and their types.  Could even detect what function names exist and only promote necessary
                    // parameters to be wide.  This would allow library implementer to add mixed varying uniform
                    // parameter versions of their functions as deemed necessary for highly used combinations
                    // versus supplying all combinations possible
                    OSL_DEV_ONLY(std::cout << "....widening value "
                                           << s.name().c_str() << std::endl);

                    OSL_ASSERT(false == function_is_uniform);
                    // As the case to deliver a pointer to a symbol data
                    // doesn't provide an opportunity to promote a uniform value
                    // to a wide value that the non-uniform function is expecting
                    // we will handle it here.

                    OSL_ASSERT(!t.is_array() && "incomplete");

                    // Have to have a place on the stack for the pointer to the wide to point to
                    llvm::Value* tmpptr
                        = getOrAllocateTemp(t, s.has_derivs(),
                                            /*is_uniform=*/false);
                    auto disable_masked_stores = ll.create_masking_scope(false);
                    int numDeriv               = s.has_derivs() ? 3 : 1;
                    for (int d = 0; d < numDeriv; ++d) {
                        for (int c = 0; c < t.simpletype().aggregate; ++c) {
                            llvm::Value* wide_value = llvm_load_value(
                                s, /*deriv=*/d, /*component*/ c,
                                TypeDesc::UNKNOWN, function_is_uniform);
                            // Store our wide pointer on the stack
                            llvm_store_value(wide_value, tmpptr, t, d, NULL, c);
                        }
                    }

                    // return pointer to our stacked wide variable
                    valargs[i] = ll.void_ptr(tmpptr);
                }
            } else {
                OSL_DEV_ONLY(std::cout << "....widening constant value "
                                       << s.name().c_str() << std::endl);

                OSL_ASSERT(s.symtype() == SymTypeConst);
                OSL_ASSERT(false == function_is_uniform);
                // As the case to deliver a pointer to a symbol data
                // doesn't provide an opportunity to promote a uniform constant
                // to a wide value that the non-uniform function is expecting.
                // So we will handle it here.

                OSL_ASSERT(!t.is_array() && "incomplete");

                OSL_ASSERT(s.has_derivs() == false
                           && "how could we have a constant with derivatives");
                // Have to have a place on the stack for the pointer to the wide constant to point to
                llvm::Value* tmpptr
                    = getOrAllocateTemp(t, false /*derivs*/,
                                        false /*function_is_uniform*/);

                auto disable_masked_stores = ll.create_masking_scope(false);
                for (int a = 0; a < t.simpletype().aggregate; ++a) {
                    llvm::Value* wide_constant_value
                        = llvm_load_constant_value(s, 0, a, TypeDesc::UNKNOWN,
                                                   function_is_uniform);
                    // Store our wide pointer on the stack
                    llvm_store_value(wide_constant_value, tmpptr, t, 0, NULL,
                                     a);
                }

                // return pointer to our stacked wide constant
                valargs[i] = ll.void_ptr(tmpptr);
            }


            OSL_DEV_ONLY(std::cout << "....pushing " << s.name().c_str()
                                   << " as void_ptr" << std::endl);
        } else {
            OSL_DEV_ONLY(std::cout << "....pushing " << s.name().c_str()
                                   << " as value" << std::endl);
            valargs[i] = llvm_load_value(s, /*deriv*/ 0, /*component*/ 0,
                                         TypeDesc::UNKNOWN,
                                         function_is_uniform);
        }
    }

    if (requiresMasking) {
        if (functionIsLlvmInlined) {
            // For inlined functions, keep the native mask type
            valargs[nargs] = ll.current_mask();
        } else {
            // For non-inlined functions, cast the mask to an int32
            valargs[nargs] = ll.mask_as_int(ll.current_mask());
        }
        // NOTE: although we accept a const FuncSpec &, we want to modify
        // its masked attribute.  As FuncSpec's are not meant to be stored
        // and should only exist on the stack to be passed into this exact
        // function (and is most likely a temporary), we don't feel bad
        // about the following const_cast, consider it by design
        const_cast<FuncSpec&>(name).mask();
    }

    OSL_DEV_ONLY(std::cout << "call_function " << build_name(name)
                           << std::endl);
    llvm::Value* func_call = ll.call_function(build_name(name), valargs);
    if (ptrToReturnStructIs1stArg) {
        if (needToBroadcastUniformResultToWide) {
            auto& wide_result_sym = *(symargs[0]);
            llvm_broadcast_uniform_value_from_mem(uniformResultTemp,
                                                  wide_result_sym);
        }
    } else {
        OSL_ASSERT(false == needToBroadcastUniformResultToWide);
    }
    return func_call;
}



llvm::Value*
BatchedBackendLLVM::llvm_call_function(const FuncSpec& name, const Symbol& A,
                                       bool deriv_ptrs)
{
    const Symbol* args[1];
    args[0] = &A;
    return llvm_call_function(name, args, 1, deriv_ptrs);
}



llvm::Value*
BatchedBackendLLVM::llvm_call_function(const FuncSpec& name, const Symbol& A,
                                       const Symbol& B, bool deriv_ptrs)
{
    const Symbol* args[2];
    args[0] = &A;
    args[1] = &B;
    return llvm_call_function(name, args, 2, deriv_ptrs);
}



llvm::Value*
BatchedBackendLLVM::llvm_call_function(const FuncSpec& name, const Symbol& A,
                                       const Symbol& B, const Symbol& C,
                                       bool deriv_ptrs,
                                       bool function_is_uniform,
                                       bool functionIsLlvmInlined,
                                       bool ptrToReturnStructIs1stArg)
{
    const Symbol* args[3];
    args[0] = &A;
    args[1] = &B;
    args[2] = &C;
    return llvm_call_function(name, args, 3, deriv_ptrs, function_is_uniform,
                              functionIsLlvmInlined, ptrToReturnStructIs1stArg);
}



llvm::Value*
BatchedBackendLLVM::llvm_test_nonzero(const Symbol& val, bool test_derivs)
{
    const TypeSpec& ts(val.typespec());
    OSL_ASSERT(!ts.is_array() && !ts.is_closure() && !ts.is_string());
    TypeDesc t = ts.simpletype();

    // Handle int case -- guaranteed no derivs, no multi-component
    if (t == TypeDesc::TypeInt) {
        // Because we allow temporaries and local results of comparison operations
        // to use the native bool type of i1, we will need to build an matching constant 0
        // for comparisons.  We can just interrogate the underlying llvm symbol to see if
        // it is a bool
        llvm::Value* llvmValue = llvm_get_pointer(val);
        //OSL_DEV_ONLY(std::cout << "llvmValue type=" << ll.llvm_typenameof(llvmValue) << std::endl);

        if (ll.llvm_typeof(llvmValue) == ll.type_ptr(ll.type_bool())) {
            return ll.op_ne(llvm_load_value(val), ll.constant_bool(0));
        } else {
            return ll.op_ne(llvm_load_value(val), ll.constant(0));
        }
    }

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
BatchedBackendLLVM::llvm_assign_impl(const Symbol& Result, const Symbol& Src,
                                     int arrayindex, int srccomp, int dstcomp)
{
    OSL_DEV_ONLY(std::cout << "llvm_assign_impl arrayindex=" << arrayindex
                           << " Result(" << Result.name() << ") is_uniform="
                           << Result.is_uniform() << std::endl);
    OSL_DEV_ONLY(std::cout << "                              Src(" << Src.name()
                           << ") is_uniform=" << Src.is_uniform() << std::endl);
    OSL_ASSERT(!Result.typespec().is_structure());
    OSL_ASSERT(!Src.typespec().is_structure());

    bool op_is_uniform = Result.is_uniform();

    const TypeSpec& result_t(Result.typespec());
    const TypeSpec& src_t(Src.typespec());

    llvm::Value* arrind = arrayindex >= 0 ? ll.constant(arrayindex) : NULL;

    if (Result.typespec().is_closure() || Src.typespec().is_closure()) {
        OSL_ASSERT(0 && "unhandled case");  // TODO: implement

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
                                           TypeDesc::FLOAT /*cast*/,
                                           op_is_uniform);
        // m=f sets the diagonal components to f, the others to zero
        llvm::Value* zero;
        if (op_is_uniform)
            zero = ll.constant(0.0f);
        else
            zero = ll.wide_constant(0.0f);

        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                llvm_store_value(i == j ? src : zero, Result, 0, arrind,
                                 i * 4 + j);
        llvm_zero_derivs(Result);  // matrices don't have derivs currently
        return true;
    }

    // memcpy complicated by promotion of uniform to wide during assignment, dissallow

    // The following code handles f=f, f=i, v=v, v=f, v=i, m=m, s=s.
    // Remember that llvm_load_value will automatically convert scalar->triple.
    TypeDesc rt              = Result.typespec().simpletype();
    TypeDesc basetype        = TypeDesc::BASETYPE(rt.basetype);
    const int num_components = rt.aggregate;
    const bool singlechan    = (srccomp != -1) || (dstcomp != -1);

    // Because we are not mem-copying arrays wholesale,
    // We will add an outer array index loop to copy 1 element or entire array
    int start_array_index = arrayindex;
    int end_array_index   = start_array_index + 1;
    if (start_array_index == -1) {
        if (result_t.is_array() && src_t.is_array()) {
            start_array_index = 0;
            end_array_index   = std::min(result_t.arraylength(),
                                       src_t.arraylength());
        }
    }
    for (arrayindex = start_array_index; arrayindex < end_array_index;
         ++arrayindex) {
        arrind = arrayindex >= 0 ? ll.constant(arrayindex) : NULL;

        if (!singlechan) {
            for (int i = 0; i < num_components; ++i) {
                // Automatically handle widening the source value to match the destination's
                llvm::Value* src_val
                    = Src.is_constant()
                          ? llvm_load_constant_value(Src, arrayindex, i,
                                                     basetype, op_is_uniform)
                          : llvm_load_value(Src, 0, arrind, i, basetype,
                                            op_is_uniform);
                if (!src_val)
                    return false;

                // The llvm_load_value above should have handled bool to int conversions
                // when the basetype == Typedesc::INT
                llvm_store_value(src_val, Result, 0, arrind, i);
            }
        } else {
            // connect individual component of an aggregate type
            // set srccomp to 0 for case when src is actually a float
            if (srccomp == -1)
                srccomp = 0;
            // Automatically handle widening the source value to match the destination's
            llvm::Value* src_val
                = Src.is_constant()
                      ? llvm_load_constant_value(Src, arrayindex, srccomp,
                                                 basetype, op_is_uniform)
                      : llvm_load_value(Src, 0, arrind, srccomp, basetype,
                                        op_is_uniform);
            if (!src_val)
                return false;

            // The llvm_load_value above should have handled bool to int conversions
            // when the basetype == Typedesc::INT

            // write source float into all compnents when dstcomp == -1, otherwise
            // the single element requested.
            if (dstcomp == -1) {
                for (int i = 0; i < num_components; ++i)
                    llvm_store_value(src_val, Result, 0, arrind, i);
            } else
                llvm_store_value(src_val, Result, 0, arrind, dstcomp);
        }

        // Handle derivatives
        if (Result.has_derivs()) {
            llvm::Value* zero = nullptr;
            if (!Src.has_derivs()) {
                // Result wants derivs but src didn't have them -- zero them
                zero = op_is_uniform ? ll.constant(0.0f)
                                     : ll.wide_constant(0.0f);
            }

            for (int d = 1; d <= 2; ++d) {
                if (!singlechan) {
                    for (int i = 0; i < num_components; ++i) {
                        llvm::Value* val = zero;
                        if (Src.has_derivs()) {
                            // src and result both have derivs -- copy them
                            // allow a uniform Src to store to a varying Result,
                            val = llvm_load_value(Src, d, arrind, i,
                                                  TypeDesc::UNKNOWN,
                                                  op_is_uniform);
                        }
                        llvm_store_value(val, Result, d, arrind, i);
                    }
                } else {
                    llvm::Value* val = zero;
                    if (Src.has_derivs()) {
                        // src and result both have derivs -- copy them
                        // allow a uniform Src to store to a varying Result,
                        val = llvm_load_value(Src, d, arrind, srccomp,
                                              TypeDesc::UNKNOWN, op_is_uniform);
                    }

                    if (dstcomp == -1) {
                        for (int i = 0; i < num_components; ++i)
                            llvm_store_value(val, Result, d, arrind, i);
                    } else
                        llvm_store_value(val, Result, d, arrind, dstcomp);
                }
            }
        }
    }
    return true;
}

int
BatchedBackendLLVM::find_userdata_index(const Symbol& sym)
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


void
BatchedBackendLLVM::append_arg_to(llvm::raw_svector_ostream& OS,
                                  const FuncSpec::Arg& arg)
{
    if (arg.is_varying()) {
        OS << m_wide_arg_prefix;
    }

    const TypeDesc& td = arg.type();
    const char* name   = nullptr;
    if (td == TypeDesc::TypeInt)
        name = "i";
    else if (td == TypeDesc::TypeMatrix)
        name = "m";
    else if (td == TypeDesc::TypeString)
        name = "s";
    else if (td == TypeDesc::TypeFloat)
        name = arg.has_derivs() ? "df" : "f";
    else if (td.aggregate == 3 && td.basetype == TypeDesc::FLOAT
             && td.arraylen == 0)
        name = arg.has_derivs() ? "dv" : "v";
    else if (td == TypeDesc(TypeDesc::PTR))
        name = "X";
    else
        OSL_ASSERT(0);
    OS << name;
}

const char*
BatchedBackendLLVM::build_name(const FuncSpec& fs)
{
    m_built_op_name.clear();

    if (fs.is_batched()) {
        OSL_ASSERT(m_library_selector != nullptr);
        if (fs.is_masked()) {
            auto op_name = llvm::Twine("osl_") + m_library_selector + fs.name();
            op_name.toVector(m_built_op_name);

            llvm::raw_svector_ostream OS(m_built_op_name);
            if (!fs.empty()) {
                OS << "_";
                for (const FuncSpec::Arg& arg : fs) {
                    append_arg_to(OS, arg);
                }
            }
            OS << "_masked";
        } else {
            auto op_name = llvm::Twine("osl_") + m_library_selector + fs.name();
            op_name.toVector(m_built_op_name);

            llvm::raw_svector_ostream OS(m_built_op_name);
            if (!fs.empty()) {
                OS << "_";
                for (const FuncSpec::Arg& arg : fs) {
                    append_arg_to(OS, arg);
                }
            }
        }
    } else {
        auto op_name = llvm::Twine("osl_") + fs.name();
        op_name.toVector(m_built_op_name);
        llvm::raw_svector_ostream OS(m_built_op_name);
        if (!fs.empty()) {
            OS << "_";
            for (const FuncSpec::Arg& arg : fs) {
                append_arg_to(OS, arg);
            }
        }
    }

    // add null terminator
    m_built_op_name.push_back(0);

    //std::cout << "Built Func Name:"  << m_built_op_name.data() << std::endl;
    return m_built_op_name.data();
}


void
BatchedBackendLLVM::llvm_print_mask(const char* title, llvm::Value* mask)
{
    llvm::Value* mask_value = ll.mask_as_int(
        (mask == nullptr) ? ll.current_mask() : mask);

    llvm::Value* call_args[] = { sg_void_ptr(),
                                 ll.constant(true_mask_value()),
                                 ll.constant("current_mask[%s]=%X (%d)\n"),
                                 ll.constant(title),
                                 mask_value,
                                 mask_value };

    ll.call_function(build_name("printf"), call_args);
}

};  // namespace pvt
OSL_NAMESPACE_EXIT
