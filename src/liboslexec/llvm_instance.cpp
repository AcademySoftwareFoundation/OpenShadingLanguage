// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <bitset>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#ifdef __GNUC__
#    include <cxxabi.h>
#endif

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/timer.h>

#include "../liboslcomp/oslcomp_pvt.h"
#include "oslexec_pvt.h"
#include "backendllvm.h"

#if OSL_USE_OPTIX
#    include <llvm/Linker/Linker.h>
#endif

// Create external declarations for all built-in funcs we may call from LLVM
#define DECL(name, signature) extern "C" void name();
#include "builtindecl.h"
#undef DECL


/*
This whole file is concerned with taking our post-optimized OSO
intermediate code and translating it into LLVM IR code so we can JIT it
and run it directly, for an expected huge speed gain over running our
interpreter.

Schematically, we want to create code that resembles the following:

    // In this example, we assume a shader group with 2 layers.

    // The GroupData struct is defined that gives the layout of the "heap",
    // the temporary memory arena that the shader can use as it executes.
    struct GroupData {
        // Array telling if we have already run each layer
        char layer_run[nlayers];
        // Array telling if we have already initialized each
        // needed user data (0 = haven't checked, 1 = checked and there
        // was no userdata, 2 = checked and there was userdata)
        char userdata_initialized[num_userdata];
        // All the user data slots, in order
        float userdata_s;
        float userdata_t;
        // For each layer in the group, we declare all shader params
        // whose values are not known -- they have init ops, or are
        // interpolated from the geom, or are connected to other layers.
        float param_0_foo;   // number is layer ID
        float param_1_bar;
    };

    // Data for the interactively adjusted parameters of this group -- these
    // can't be turned into constants because the app may want to modify them
    // as it runs (such as for user interaction). This block of memory has
    // one global copy specific each the shader group, managed by OSL.
    struct InteractiveParams {
        float iparam_0_baz;
    };

    // Name of layer entry is $layer_ID
    void $layer_0(ShaderGlobals* sg, GroupData* group,
                  void* userdatda_base_ptr, void* output_base_ptr,
                  int shadeindex, InteractiveParams* interactive_params)
    {
        // Declare locals, temps, constants, params with known values.
        // Make them all look like stack memory locations:
        float *x = alloca(sizeof(float));
        // ...and so on for all the other locals & temps...

        // then run the shader body:
        *x = sg->u * group->param_0_bar;
        group->param_1_foo = *x;
        *x += interactive_params->iparam_0_baz;
        // ...
    }

    void $layer_1(ShaderGlobals* sg, GroupData* group,
                  void* userdatda_base_ptr, void* output_base_ptr,
                  int shadeindex, InteractiveParams* interactive_params)
    {
        // Because we need the outputs of layer 0 now, we call it if it
        // hasn't already run:
        if (! group->layer_run[0]) {
            group->layer_run[0] = 1;
            $layer_0 (sg, group, userdata_base_ptr, output_base_ptr,
                     shadeindex, interactive_params); // because we need its outputs
        }
        *y = sg->u * group->$param_1_bar;
    }

    void $group_1(ShaderGlobals* sg, GroupData* group,
                  void* userdatda_base_ptr, void* output_base_ptr,
                  int shadeindex, InteractiveParams* interactive_params)
    {
        group->layer_run[...] = 0;
        // Run just the unconditional layers

        if (! group->layer_run[1]) {
            group->layer_run[1] = 1;
            $layer_1(sg, group, userdata_base_ptr, output_base_ptr,
                     shadeindex, interactive_params);
        }
    }

*/

extern int osl_llvm_compiled_ops_size;
extern unsigned char osl_llvm_compiled_ops_block[];

extern int osl_llvm_compiled_rs_dependent_ops_size;
extern unsigned char osl_llvm_compiled_rs_dependent_ops_block[];

#ifdef OSL_LLVM_CUDA_BITCODE
extern int shadeops_cuda_llvm_compiled_ops_size;
extern unsigned char shadeops_cuda_llvm_compiled_ops_block[];
#endif

using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace pvt {

static spin_mutex llvm_mutex;

static ustring op_end("end");
static ustring op_nop("nop");
static ustring op_aassign("aassign");
static ustring op_compassign("compassign");
static ustring op_mxcompassign("mxcompassign");
static ustring op_aref("aref");
static ustring op_compref("compref");
static ustring op_mxcompref("mxcompref");
static ustring op_useparam("useparam");
static ustring unknown_shader_group_name("<Unknown Shader Group Name>");


struct HelperFuncRecord {
    const char* argtypes;
    void (*function)();
    HelperFuncRecord(const char* argtypes = NULL, void (*function)() = NULL)
        : argtypes(argtypes), function(function)
    {
    }
};

typedef std::unordered_map<std::string, HelperFuncRecord> HelperFuncMap;
static HelperFuncMap llvm_helper_function_map;
static atomic_int llvm_helper_function_map_initialized(0);
static spin_mutex llvm_helper_function_map_mutex;
static std::vector<std::string>
    external_function_names;  // used for internalize_module_functions



static void
initialize_llvm_helper_function_map()
{
    if (llvm_helper_function_map_initialized)
        return;  // already done
    spin_lock lock(llvm_helper_function_map_mutex);
    if (llvm_helper_function_map_initialized)
        return;
#define DECL(name, signature)                                            \
    llvm_helper_function_map[#name] = HelperFuncRecord(signature, name); \
    external_function_names.push_back(#name);
#include "builtindecl.h"
#undef DECL

    llvm_helper_function_map_initialized = 1;
}



static void*
helper_function_lookup(const std::string& name)
{
    HelperFuncMap::const_iterator i = llvm_helper_function_map.find(name);
    if (i == llvm_helper_function_map.end())
        return NULL;
    return (void*)i->second.function;
}


std::string
layer_function_name(const ShaderGroup& group, const ShaderInstance& inst,
                    bool api)
{
    bool use_optix     = inst.shadingsys().use_optix();
    const char* prefix = use_optix && api ? "__direct_callable__" : "";
    return fmtformat("{}osl_layer_group_{}_name_{}", prefix, group.name(),
                     inst.layername());
}

std::string
init_function_name(const ShadingSystemImpl& shadingsys,
                   const ShaderGroup& group, bool api)
{
    bool use_optix     = shadingsys.use_optix();
    const char* prefix = use_optix && api ? "__direct_callable__" : "";

    return fmtformat("{}osl_init_group_{}", prefix, group.name());
}

std::string
fused_function_name(const ShaderGroup& group)
{
    int nlayers          = group.nlayers();
    ShaderInstance* inst = group[nlayers - 1];

    return fmtformat("__direct_callable__fused_{}_name_{}", group.name(),
                     inst->layername());
}

llvm::Type*
BackendLLVM::llvm_type_sg()
{
    // Create a type that defines the ShaderGlobals for LLVM IR.  This
    // absolutely MUST exactly match the ShaderGlobals struct in oslexec.h.
    if (m_llvm_type_sg)
        return m_llvm_type_sg;

    // Derivs look like arrays of 3 values
    llvm::Type* float_deriv = llvm_type(
        TypeDesc(TypeDesc::FLOAT, TypeDesc::SCALAR, 3));
    llvm::Type* triple_deriv = llvm_type(
        TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, 3));
    std::vector<llvm::Type*> sg_types;
    sg_types.push_back(triple_deriv);      // P, dPdx, dPdy
    sg_types.push_back(ll.type_triple());  // dPdz
    sg_types.push_back(triple_deriv);      // I, dIdx, dIdy
    sg_types.push_back(ll.type_triple());  // N
    sg_types.push_back(ll.type_triple());  // Ng
    sg_types.push_back(float_deriv);       // u, dudx, dudy
    sg_types.push_back(float_deriv);       // v, dvdx, dvdy
    sg_types.push_back(ll.type_triple());  // dPdu
    sg_types.push_back(ll.type_triple());  // dPdv
    sg_types.push_back(ll.type_float());   // time
    sg_types.push_back(ll.type_float());   // dtime
    sg_types.push_back(ll.type_triple());  // dPdtime
    sg_types.push_back(triple_deriv);      // Ps

    llvm::Type* vp = (llvm::Type*)ll.type_void_ptr();
    sg_types.push_back(vp);             // opaque renderstate*
    sg_types.push_back(vp);             // opaque tracedata*
    sg_types.push_back(vp);             // opaque objdata*
    sg_types.push_back(vp);             // ShadingContext*
    sg_types.push_back(vp);             // OpaqueShadingStateUniformPtr
    sg_types.push_back(ll.type_int());  //thread_index
    sg_types.push_back(ll.type_int());  //shade_index
    sg_types.push_back(vp);             // RendererServices*
    sg_types.push_back(vp);             // object2common
    sg_types.push_back(vp);             // shader2common
    sg_types.push_back(vp);             // Ci

    sg_types.push_back(ll.type_float());  // surfacearea
    sg_types.push_back(ll.type_int());    // raytype
    sg_types.push_back(ll.type_int());    // flipHandedness
    sg_types.push_back(ll.type_int());    // backfacing

    return m_llvm_type_sg = ll.type_struct(sg_types, "ShaderGlobals");
}



llvm::Type*
BackendLLVM::llvm_type_sg_ptr()
{
    return ll.type_ptr(llvm_type_sg());
}



llvm::Type*
BackendLLVM::llvm_type_groupdata()
{
    // If already computed, return it
    if (m_llvm_type_groupdata)
        return m_llvm_type_groupdata;

    std::vector<llvm::Type*> fields;
    int offset = 0;
    int order  = 0;
    m_groupdata_field_names.clear();

    if (llvm_debug() >= 2)
        std::cout << "Group param struct:\n";

    // First, add the array that tells if each layer has run.  But only make
    // slots for the layers that may be called/used.
    if (llvm_debug() >= 2)
        std::cout << "  layers run flags: " << m_num_used_layers
                  << " at offset " << offset << "\n";
    int sz = (m_num_used_layers + 3) & (~3);  // Round up to 32 bit boundary
    fields.push_back(ll.type_array(ll.type_bool(), sz));
    m_groupdata_field_names.emplace_back("layer_runflags");
    offset += sz * sizeof(bool);
    ++order;

    // Now add the array that tells which userdata have been initialized,
    // and the space for the userdata values.
    int nuserdata = (int)group().m_userdata_names.size();
    if (nuserdata) {
        if (llvm_debug() >= 2)
            std::cout << "  userdata initialized flags: " << nuserdata
                      << " at offset " << offset << ", field " << order << "\n";
        ustring* names  = &group().m_userdata_names[0];
        TypeDesc* types = &group().m_userdata_types[0];
        int* offsets    = &group().m_userdata_offsets[0];
        int sz          = (nuserdata + 3) & (~3);
        fields.push_back(ll.type_array(ll.type_bool(), sz));
        m_groupdata_field_names.emplace_back("userdata_init_flags");
        offset += nuserdata * sizeof(bool);
        ++order;
        for (int i = 0; i < nuserdata; ++i) {
            TypeDesc type = types[i];
            // make room for float derivs only
            type.arraylen = type.basetype == TypeDesc::FLOAT
                                ? type.numelements() * 3
                                : type.numelements();
            fields.push_back(llvm_type(type));
            m_groupdata_field_names.emplace_back(
                fmtformat("userdata{}_{}_", i, names[i]));
            // Alignment
            int align = type.basesize();
            offset    = OIIO::round_to_multiple_of_pow2(offset, align);
            if (llvm_debug() >= 2)
                std::cout << "  userdata " << names[i] << ' ' << type
                          << ", field " << order << ", offset " << offset
                          << "\n";
            offsets[i] = offset;
            offset += int(type.size());
            ++order;
        }
    }

    // For each layer in the group, add entries for all params that are
    // connected or interpolated, and output params.  Also mark those
    // symbols with their offset within the group struct.
    m_param_order_map.clear();
    for (int layer = 0; layer < group().nlayers(); ++layer) {
        ShaderInstance* inst = group()[layer];
        if (inst->unused())
            continue;
        FOREACH_PARAM(Symbol & sym, inst)
        {
            TypeSpec ts = sym.typespec();
            if (ts.is_structure())  // skip the struct symbol itself
                continue;

            if (can_treat_param_as_local(sym))
                continue;

            const int arraylen  = std::max(1, sym.typespec().arraylength());
            const int derivSize = (sym.has_derivs() ? 3 : 1);
            ts.make_array(arraylen * derivSize);
            fields.push_back(llvm_type(ts));
            m_groupdata_field_names.emplace_back(
                fmtformat("lay{}param_{}_", layer, sym.name()));

            // FIXME(arena) -- temporary debugging
            if (debug() && sym.symtype() == SymTypeOutputParam
                && !sym.connected_down()) {
                auto found = group().find_symloc(sym.name());
                if (found)
                    print("layer {} \"{}\" : OUTPUT {}\n", layer,
                          inst->layername(), found->name);
            }

            // Alignment
            size_t align = sym.typespec().is_closure_based()
                               ? sizeof(void*)
                               : sym.typespec().simpletype().basesize();
            if (offset & (align - 1))
                offset += align - (offset & (align - 1));
            if (llvm_debug() >= 2)
                print("  {} ({}) {} {}, field {}, size {}, offset {}{}{}\n",
                      inst->layername(), inst->id(), sym.mangled(), ts.c_str(),
                      order, derivSize * int(sym.size()), offset,
                      sym.interpolated() ? " (interpolated)" : "",
                      sym.interactive() ? " (interactive)" : "");
            sym.dataoffset((int)offset);
            // TODO(arenas): sym.set_dataoffset(SymArena::Heap, offset);
            offset += derivSize * sym.size();
            m_param_order_map[&sym] = order;
            ++order;
        }
    }
    group().llvm_groupdata_size(offset);
    if (llvm_debug() >= 2)
        print(" Group struct had {} fields, total size {}\n\n", order, offset);

    m_llvm_type_groupdata = ll.type_struct(fields, "Groupdata");
    OSL_ASSERT(fields.size() == m_groupdata_field_names.size());

    return m_llvm_type_groupdata;
}



llvm::Type*
BackendLLVM::llvm_type_groupdata_ptr()
{
    return ll.type_ptr(llvm_type_groupdata());
}



llvm::Type*
BackendLLVM::llvm_type_closure_component()
{
    if (m_llvm_type_closure_component)
        return m_llvm_type_closure_component;

    std::vector<llvm::Type*> comp_types;
    comp_types.push_back(ll.type_int());     // id
    comp_types.push_back(ll.type_triple());  // w
    comp_types.push_back(ll.type_int());     // fake field for char mem[4]

    return m_llvm_type_closure_component = ll.type_struct(comp_types,
                                                          "ClosureComponent");
}



llvm::Type*
BackendLLVM::llvm_type_closure_component_ptr()
{
    return ll.type_ptr(llvm_type_closure_component());
}

void
BackendLLVM::build_offsets_of_ShaderGlobals(
    std::vector<unsigned int>& offset_by_index)
{
    offset_by_index.push_back(offsetof(ShaderGlobals, P));
    offset_by_index.push_back(offsetof(ShaderGlobals, dPdz));
    offset_by_index.push_back(offsetof(ShaderGlobals, I));
    offset_by_index.push_back(offsetof(ShaderGlobals, N));
    offset_by_index.push_back(offsetof(ShaderGlobals, Ng));
    offset_by_index.push_back(offsetof(ShaderGlobals, u));
    offset_by_index.push_back(offsetof(ShaderGlobals, v));


    offset_by_index.push_back(offsetof(ShaderGlobals, dPdu));
    offset_by_index.push_back(offsetof(ShaderGlobals, dPdv));
    offset_by_index.push_back(offsetof(ShaderGlobals, time));
    offset_by_index.push_back(offsetof(ShaderGlobals, dtime));
    offset_by_index.push_back(offsetof(ShaderGlobals, dPdtime));
    offset_by_index.push_back(offsetof(ShaderGlobals, Ps));


    offset_by_index.push_back(offsetof(ShaderGlobals, renderstate));
    offset_by_index.push_back(offsetof(ShaderGlobals, tracedata));
    offset_by_index.push_back(offsetof(ShaderGlobals, objdata));
    offset_by_index.push_back(offsetof(ShaderGlobals, context));
    offset_by_index.push_back(offsetof(ShaderGlobals, shadingStateUniform));

    offset_by_index.push_back(offsetof(ShaderGlobals, thread_index));
    offset_by_index.push_back(offsetof(ShaderGlobals, shade_index));

    offset_by_index.push_back(offsetof(ShaderGlobals, renderer));

    offset_by_index.push_back(offsetof(ShaderGlobals, object2common));
    offset_by_index.push_back(offsetof(ShaderGlobals, shader2common));
    offset_by_index.push_back(offsetof(ShaderGlobals, Ci));

    offset_by_index.push_back(offsetof(ShaderGlobals, surfacearea));
    offset_by_index.push_back(offsetof(ShaderGlobals, raytype));
    offset_by_index.push_back(offsetof(ShaderGlobals, flipHandedness));
    offset_by_index.push_back(offsetof(ShaderGlobals, backfacing));
}



void
BackendLLVM::llvm_assign_initial_value(const Symbol& sym, bool force)
{
    // Don't write over connections!  Connection values are written into
    // our layer when the earlier layer is run, as part of its code.  So
    // we just don't need to initialize it here at all.
    if (!force && sym.valuesource() == Symbol::ConnectedVal
        && !sym.typespec().is_closure_based())
        return;
    // For "globals" that are closures, there is nothing to initialize.
    if (sym.typespec().is_closure_based() && sym.symtype() == SymTypeGlobal)
        return;

    // Closures need to get their storage before anything can be
    // assigned to them.  Unless they are params, in which case we took
    // care of it in the group entry point.
    if (sym.typespec().is_closure_based() && sym.symtype() != SymTypeParam
        && sym.symtype() != SymTypeOutputParam) {
        llvm_assign_zero(sym);
        return;
    }

    // For local variables (including temps), when "debug_uninit" is enabled,
    // we store special values in the variable to make it easier to detect
    // uninitialized use.
    if ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp)
        && shadingsys().debug_uninit()) {
        bool isarray   = sym.typespec().is_array();
        int alen       = isarray ? sym.typespec().arraylength() : 1;
        llvm::Value* u = NULL;
        if (sym.typespec().is_closure_based()) {
            // skip closures
        } else if (sym.typespec().is_float_based())
            u = ll.constant(std::numeric_limits<float>::quiet_NaN());
        else if (sym.typespec().is_int_based())
            u = ll.constant(std::numeric_limits<int>::min());
        else if (sym.typespec().is_string_based())
            u = llvm_load_string(Strings::uninitialized_string);
        if (u) {
            for (int a = 0; a < alen; ++a) {
                llvm::Value* aval = isarray ? ll.constant(a) : NULL;
                for (int c = 0; c < (int)sym.typespec().aggregate(); ++c)
                    llvm_store_value(u, sym, 0, aval, c);
            }
        }
        return;
    }

    // Local/temp strings are always initialized to the empty string.
    if ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp)
        && sym.typespec().is_string_based()) {
        // Strings are pointers.  Can't take any chance on leaving
        // local/tmp syms uninitialized.
        llvm_assign_zero(sym);
        return;  // we're done, the parts below are just for params
    }

    // FIXME: Future work -- how expensive would it be to initialize all
    // locals to 0? We should test this for performance hit, and if
    // reasonable, it may be a good idea to do it by default.

    // From here on, everything we are dealing with is a shader parameter
    // (either ordinary or output).
    OSL_ASSERT_MSG(sym.symtype() == SymTypeParam
                       || sym.symtype() == SymTypeOutputParam,
                   "symtype was %d, data type was %s", (int)sym.symtype(),
                   sym.typespec().c_str());

    // Handle interpolated params by calling osl_bind_interpolated_param,
    // which will check if userdata is already retrieved, if not it will
    // call RendererServices::get_userdata to retrieve it. In either case,
    // it will return 1 if it put the userdata in the right spot (either
    // retrieved de novo or copied from a previous retrieval), or 0 if no
    // such userdata was available.
    llvm::BasicBlock* after_userdata_block = nullptr;
    const SymLocationDesc* symloc          = nullptr;
    if (sym.interpolated() && !sym.typespec().is_closure()) {
        ustring symname = sym.name();
        TypeDesc type   = sym.typespec().simpletype();

        int userdata_index = find_userdata_index(sym);
        OSL_DASSERT(userdata_index >= 0);

        llvm::Value* got_userdata = nullptr;

        // See if userdata input placement has been used for this symbol
        ustring layersym = ustring::fmtformat("{}.{}", inst()->layername(),
                                              sym.name());
        symloc           = group().find_symloc(layersym, SymArena::UserData);
        if (!symloc)
            symloc = group().find_symloc(sym.name(), SymArena::UserData);
        if (symloc) {
            // We had a userdata pre-placement record for this variable.
            // Just copy from the correct offset location!

            // Strutil::print("GEN found placeable userdata input {} -> {} {} size={}\n",
            //                sym.name(), symloc->name, sym.typespec(),
            //                symloc->type.size());
            int size = int(symloc->type.size());
            if (symloc->derivs && sym.has_derivs())
                size *= 3;  // If we're copying the derivs
            llvm::Value* srcptr = symloc_ptr(symloc, m_llvm_userdata_base_ptr);
            llvm::Value* dstptr = llvm_void_ptr(sym);
            ll.op_memcpy(dstptr, srcptr, size);
            // Clear derivs if the variable wants derivs but placement
            // source didn't have them.
            if (sym.has_derivs() && !symloc->derivs)
                ll.op_memset(ll.offset_ptr(dstptr, size), 0, 2 * size);
        } else {
            // No pre-placement: fall back to call to the renderer callback.
            llvm::Value* args[] = {
                sg_void_ptr(),
                llvm_load_string(symname),
                ll.constant(type),
                ll.constant((int)group().m_userdata_derivs[userdata_index]),
                groupdata_field_ptr(2 + userdata_index),  // userdata data ptr
                ll.constant((int)sym.has_derivs()),
                llvm_void_ptr(sym),
                ll.constant(sym.derivsize()),
                ll.void_ptr(userdata_initialized_ref(userdata_index)),
                ll.constant(userdata_index),
            };
            got_userdata = ll.call_function("osl_bind_interpolated_param",
                                            args);
        }
        if (shadingsys().debug_nan() && type.basetype == TypeDesc::FLOAT) {
            // check for NaN/Inf for float-based types
            int ncomps          = type.numelements() * type.aggregate;
            llvm::Value* args[] = { ll.constant(ncomps),
                                    llvm_void_ptr(sym),
                                    ll.constant((int)sym.has_derivs()),
                                    sg_void_ptr(),
                                    llvm_load_stringhash(inst()->shadername()),
                                    ll.constant(0),
                                    llvm_load_stringhash(sym.unmangled()),
                                    ll.constant(0),
                                    ll.constant(ncomps),
                                    llvm_load_stringhash("<get_userdata>") };
            ll.call_function("osl_naninf_check", args);
        }
        // userdata pre-placement always succeeds, we don't need to bother
        // with handing partial results possibly from bind_interpolated_param
        if (symloc == nullptr) {
            // We will enclose the subsequent initialization of default values
            // or init ops in an "if" so that the extra copies or code don't
            // happen if the userdata was retrieved.
            llvm::BasicBlock* no_userdata_block = ll.new_basic_block(
                "no_userdata");
            after_userdata_block  = ll.new_basic_block();
            llvm::Value* cond_val = ll.op_eq(got_userdata, ll.constant(0));
            ll.op_branch(cond_val, no_userdata_block, after_userdata_block);
        }
    }

    // Only generate init_ops or default assignment when userdata pre-placement
    // is not found
    if (symloc == nullptr) {
        if (sym.has_init_ops() && sym.valuesource() == Symbol::DefaultVal) {
            // Handle init ops.
            build_llvm_code(sym.initbegin(), sym.initend());
#if OSL_USE_OPTIX
        } else if (use_optix() && !sym.typespec().is_closure()
                   && !sym.lockgeom()) {
            // If the call to osl_bind_interpolated_param returns 0, the default
            // value needs to be loaded from a CUDA variable.
            llvm::Value* cuda_var     = getOrAllocateCUDAVariable(sym);
            TypeSpec elemtype         = sym.typespec().elementtype();
            llvm::Type* cuda_var_type = llvm_type(elemtype);
            // memcpy the initial value from the CUDA variable
            llvm::Value* src = ll.ptr_cast(ll.GEP(cuda_var_type, cuda_var, 0),
                                           ll.type_void_ptr());
            llvm::Value* dst = llvm_void_ptr(sym);
            TypeDesc t       = sym.typespec().simpletype();
            ll.op_memcpy(dst, src, t.size(), t.basesize());
            if (sym.has_derivs())
                llvm_zero_derivs(sym);
#endif
        } else if (sym.interpolated() && !sym.typespec().is_closure()) {
            // geometrically-varying param; memcpy its default value
            TypeDesc t = sym.typespec().simpletype();
            ll.op_memcpy(llvm_void_ptr(sym), ll.constant_ptr(sym.data()),
                         t.size(), t.basesize() /*align*/);
            if (sym.has_derivs())
                llvm_zero_derivs(sym);
        } else {
            // Use default value
            int num_components = sym.typespec().simpletype().aggregate;
            TypeSpec elemtype  = sym.typespec().elementtype();
            int arraylen       = std::max(1, sym.typespec().arraylength());
            for (int a = 0, c = 0; a < arraylen; ++a) {
                llvm::Value* arrind = sym.typespec().is_array() ? ll.constant(a)
                                                                : NULL;
                if (sym.typespec().is_closure_based())
                    continue;
                for (int i = 0; i < num_components; ++i, ++c) {
                    // Fill in the constant val
                    llvm::Value* init_val = 0;
                    if (elemtype.is_float_based())
                        init_val = ll.constant(sym.get_float(c));
                    else if (elemtype.is_string())
                        init_val = llvm_load_string(sym.get_string(c));
                    else if (elemtype.is_int())
                        init_val = ll.constant(sym.get_int(c));
                    OSL_DASSERT(init_val);
                    llvm_store_value(init_val, sym, 0, arrind, i);
                }
            }
            if (sym.has_derivs())
                llvm_zero_derivs(sym);
        }

        if (after_userdata_block) {
            // If we enclosed the default initialization in an "if", jump to the
            // next basic block now.
            ll.op_branch(after_userdata_block);
        }
    }
}



void
BackendLLVM::llvm_generate_debugnan(const Opcode& op)
{
    // This function inserts extra debugging code to make sure that this op
    // did not produce any NaN values.

    // Check each argument to the op...
    for (int i = 0; i < op.nargs(); ++i) {
        // Only consider the arguments that this op WRITES to
        if (!op.argwrite(i))
            continue;

        Symbol& sym(*opargsym(op, i));
        TypeDesc t = sym.typespec().simpletype();

        // Only consider floats, because nothing else can be a NaN
        if (t.basetype != TypeDesc::FLOAT)
            continue;

        // Default: Check all elements of the variable being written
        llvm::Value* ncomps = ll.constant(int(t.numelements() * t.aggregate));
        llvm::Value* offset = ll.constant(0);
        llvm::Value* ncheck = ncomps;

        // There are a few special cases where an op writes a partial value:
        // one element of an array, aggregate (like a point), or matrix. In
        // those cases, don't check the other pieces that haven't been
        // touched by this op, because (a) it's unnecessary work and code
        // generation, and (b) they might generate a false positive error.
        // An example would be:
        //     float A[3];     // 1   Elements could be anything
        //     A[1] = x;       // 2   <-- this is where we are
        // Line 2 only wrote element [1], so we do not need to check the
        // current values of [0] or [2].
        if (op.opname() == op_aassign) {
            OSL_DASSERT(i == 0 && "only arg 0 is written for aassign");
            llvm::Value* ind = llvm_load_value(*opargsym(op, 1));
            llvm::Value* agg = ll.constant(t.aggregate);
            offset           = t.aggregate == 1 ? ind : ll.op_mul(ind, agg);
            ncheck           = agg;
        } else if (op.opname() == op_compassign) {
            OSL_DASSERT(i == 0 && "only arg 0 is written for compassign");
            llvm::Value* ind = llvm_load_value(*opargsym(op, 1));
            offset           = ind;
            ncheck           = ll.constant(1);
        } else if (op.opname() == op_mxcompassign) {
            OSL_DASSERT(i == 0 && "only arg 0 is written for mxcompassign");
            Symbol& row_sym      = *opargsym(op, 1);
            Symbol& col_sym      = *opargsym(op, 2);
            llvm::Value* row_ind = llvm_load_value(row_sym);
            llvm::Value* col_ind = llvm_load_value(col_sym);
            llvm::Value* comp    = ll.op_mul(row_ind, ll.constant(4));
            comp                 = ll.op_add(comp, col_ind);
            offset               = comp;
            ncheck               = ll.constant(1);
        }

        llvm::Value* args[] = { ncomps,
                                llvm_void_ptr(sym),
                                ll.constant((int)sym.has_derivs()),
                                sg_void_ptr(),
                                llvm_load_stringhash(op.sourcefile()),
                                ll.constant(op.sourceline()),
                                llvm_load_stringhash(sym.unmangled()),
                                offset,
                                ncheck,
                                llvm_load_stringhash(op.opname()) };
        ll.call_function("osl_naninf_check", args);
    }
}



void
BackendLLVM::llvm_generate_debug_uninit(const Opcode& op)
{
    // This function inserts extra debugging code to make sure that this op
    // did not read an uninitialized value.

    if (op.opname() == op_useparam) {
        // Don't check the args of a useparam before the op; they are by
        // definition potentially not yet set before the useparam action
        // itself puts values into them. Checking them for uninitialized
        // values will result in false positives.
        return;
    }

    // Check each argument to the op...
    for (int i = 0; i < op.nargs(); ++i) {
        // Only consider the arguments that this op READS
        if (!op.argread(i))
            continue;

        Symbol& sym(*opargsym(op, i));

        // just check float, int, string based types.
        if (sym.typespec().is_closure_based())
            continue;
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT && t.basetype != TypeDesc::INT
            && t.basetype != TypeDesc::STRING)
            continue;

        // Some special cases...
        if (op.opname() == Strings::op_for && i == 0) {
            // The first argument of 'for' is the condition temp, but
            // note that it may not have had its initializer run yet, so
            // don't generate uninit test code for it.
            continue;
        }
        if (op.opname() == Strings::op_dowhile && i == 0) {
            // The first argument of 'dowhile' is the condition temp, but
            // most likely its initializer has not run yet. Unless there is
            // no "condition" code block, in that case we should still test
            // it for uninit.
            if (op.jump(0) != op.jump(1))
                continue;
        }

        // Default: Check all elements of the variable being read
        llvm::Value* ncheck = ll.constant(int(t.numelements() * t.aggregate));
        llvm::Value* offset = ll.constant(0);

        // There are a few special cases where an op reads a partial value:
        // one element of an array, aggregate (like a point), or matrix. In
        // those cases, don't check the other pieces that haven't been
        // touched by this op, because (a) it's unnecessary work and code
        // generation, and (b) they might generate a false positive error.
        // An example would be:
        //     float A[3];     // 1   Elements are uninitialized
        //     A[1] = 1;       // 2
        //     float x = A[1]; // 3   <--- this is where we are
        // Line 3 only reads element [1]. It is NOT an uninitialized read,
        // even though other parts of array A are uninitialized. And even if
        // they were initialized, it's unnecessary to check the status of
        // the other elements that we didn't just read. Even if [2] is
        // uninitialized for the whole shader, that doesn't matter as long
        // as we don't try to read it.
        if (op.opname() == op_aref && i == 1) {
            // Special case -- array assignment -- only check one element
            llvm::Value* ind = llvm_load_value(*opargsym(op, 2));
            llvm::Value* agg = ll.constant(t.aggregate);
            offset           = t.aggregate == 1 ? ind : ll.op_mul(ind, agg);
            ncheck           = agg;
        } else if (op.opname() == op_compref && i == 1) {
            // Special case -- component assignment -- only check one channel
            llvm::Value* ind = llvm_load_value(*opargsym(op, 2));
            offset           = ind;
            ncheck           = ll.constant(1);
        } else if (op.opname() == op_mxcompref && i == 1) {
            // Special case -- matrix component reference -- only check one channel
            Symbol& row_sym      = *opargsym(op, 2);
            Symbol& col_sym      = *opargsym(op, 3);
            llvm::Value* row_ind = llvm_load_value(row_sym);
            llvm::Value* col_ind = llvm_load_value(col_sym);

            llvm::Value* comp = ll.op_mul(row_ind, ll.constant(4));
            comp              = ll.op_add(comp, col_ind);
            offset            = comp;
            ncheck            = ll.constant(1);
        }

        llvm::Value* args[] = { ll.constant(t),
                                llvm_void_ptr(sym),
                                sg_void_ptr(),
                                llvm_load_stringhash(op.sourcefile()),
                                ll.constant(op.sourceline()),
                                llvm_load_stringhash(group().name()),
                                ll.constant(layer()),
                                llvm_load_stringhash(inst()->layername()),
                                llvm_load_stringhash(inst()->shadername()),
                                ll.constant(int(&op - &inst()->ops()[0])),
                                llvm_load_stringhash(op.opname()),
                                ll.constant(i),
                                llvm_load_stringhash(sym.unmangled()),
                                offset,
                                ncheck };
        ll.call_function("osl_uninit_check", args);
    }
}



void
BackendLLVM::llvm_generate_debug_op_printf(const Opcode& op)
{
    std::ostringstream msg;
    msg.imbue(std::locale::classic());  // force C locale
    msg << op.sourcefile() << ':' << op.sourceline() << ' ' << op.opname();
    for (int i = 0; i < op.nargs(); ++i)
        msg << ' ' << opargsym(op, i)->mangled();
    llvm_gen_debug_printf(msg.str());
}



bool
BackendLLVM::build_llvm_code(int beginop, int endop, llvm::BasicBlock* bb)
{
    if (bb)
        ll.set_insert_point(bb);

    for (int opnum = beginop; opnum < endop; ++opnum) {
        const Opcode& op        = inst()->ops()[opnum];
        const OpDescriptor* opd = shadingsys().op_descriptor(op.opname());
        if (opd && opd->llvmgen) {
            if (shadingsys().debug_uninit() /* debug uninitialized vals */)
                llvm_generate_debug_uninit(op);
            if (shadingsys().llvm_debug_ops())
                llvm_generate_debug_op_printf(op);
            if (ll.debug_is_enabled())
                ll.debug_set_location(op.sourcefile(),
                                      std::max(op.sourceline(), 1));
            bool ok = (*opd->llvmgen)(*this, opnum);
            if (!ok)
                return false;
            if (shadingsys().debug_nan() /* debug NaN/Inf */
                && op.farthest_jump() < 0 /* Jumping ops don't need it */) {
                llvm_generate_debugnan(op);
            }
        } else if (op.opname() == op_nop || op.opname() == op_end) {
            // Skip this op, it does nothing...
        } else {
            shadingcontext()->errorfmt(
                "LLVMOSL: Unsupported op {} in layer {}\n", op.opname(),
                inst()->layername());
            return false;
        }

        // If the op we coded jumps around, skip past its recursive block
        // executions.
        int next = op.farthest_jump();
        if (next >= 0)
            opnum = next - 1;
    }
    return true;
}



llvm::Function*
BackendLLVM::build_llvm_init()
{
    // Make a group init function: void group_init(ShaderGlobals*, GroupData*)
    // Note that the GroupData* is passed as a void*.
    std::string unique_name = init_function_name(shadingsys(), group());
    ll.current_function(
        ll.make_function(unique_name, false,
                         ll.type_void(),  // return type
                         {
                             llvm_type_sg_ptr(), llvm_type_groupdata_ptr(),
                             ll.type_void_ptr(),  // userdata_base_ptr
                             ll.type_void_ptr(),  // output_base_ptr
                             ll.type_int(),
                             ll.type_void_ptr(),  // FIXME: interactive params
                         }));

    if (ll.debug_is_enabled()) {
        ustring sourcefile
            = group()[0]->op(group()[0]->maincodebegin()).sourcefile();
        ll.debug_push_function(unique_name, sourcefile, 0);
    }

    // Get shader globals and groupdata pointers
    m_llvm_shaderglobals_ptr = ll.current_function_arg(0);  //arg_it++;
    m_llvm_shaderglobals_ptr->setName("shaderglobals_ptr");
    m_llvm_groupdata_ptr = ll.current_function_arg(1);  //arg_it++;
    m_llvm_groupdata_ptr->setName("groupdata_ptr");
    m_llvm_userdata_base_ptr = ll.current_function_arg(2);  //arg_it++;
    m_llvm_userdata_base_ptr->setName("userdata_base_ptr");
    m_llvm_output_base_ptr = ll.current_function_arg(3);  //arg_it++;
    m_llvm_output_base_ptr->setName("output_base_ptr");
    m_llvm_shadeindex = ll.current_function_arg(4);  //arg_it++;
    m_llvm_shadeindex->setName("shadeindex");
    m_llvm_interactive_params_ptr = ll.current_function_arg(5);  //arg_it++;
    m_llvm_interactive_params_ptr->setName("interactive_params_ptr");

    // Set up a new IR builder
    llvm::BasicBlock* entry_bb = ll.new_basic_block(unique_name);
    ll.new_builder(entry_bb);
#if 0 /* helpful for debugging */
    if (llvm_debug()) {
        llvm_gen_debug_printf (fmtformat("\n\n\n\nGROUP! {}",group().name()));
        llvm_gen_debug_printf ("enter group initlayer %d %s %s",
                               this->layer(), inst()->layername(), inst()->shadername()));
    }
#endif

    // Group init clears all the "layer_run" and "userdata_initialized" flags.
    if (m_num_used_layers > 1) {
        int sz = (m_num_used_layers + 3) & (~3);  // round up to 32 bits
        ll.op_memset(ll.void_ptr(layer_run_ref(0)), 0, sz, 4 /*align*/);
    }
    int num_userdata = (int)group().m_userdata_names.size();
    if (num_userdata) {
        int sz = (num_userdata + 3) & (~3);  // round up to 32 bits
        ll.op_memset(ll.void_ptr(userdata_initialized_ref(0)), 0, sz,
                     4 /*align*/);
    }

    // Group init also needs to allot space for ALL layers' params
    // that are closures (to avoid weird order of layer eval problems).
    for (int i = 0; i < group().nlayers(); ++i) {
        ShaderInstance* gi = group()[i];
        if (gi->unused() || gi->empty_instance())
            continue;
        FOREACH_PARAM(Symbol & sym, gi)
        {
            if (sym.typespec().is_closure_based()) {
                int arraylen     = std::max(1, sym.typespec().arraylength());
                llvm::Value* val = ll.constant_ptr(NULL, ll.type_void_ptr());
                for (int a = 0; a < arraylen; ++a) {
                    llvm::Value* arrind = sym.typespec().is_array()
                                              ? ll.constant(a)
                                              : NULL;
                    llvm_store_value(val, sym, 0, arrind, 0);
                }
            }
        }
    }


    // All done
#if 0 /* helpful for debugging */
    if (llvm_debug())
        llvm_gen_debug_printf(fmtformat("exit group init {}", group().name());
#endif
    ll.op_return();

    if (llvm_debug())
        print("group init func ({}) after llvm  = {}\n", unique_name,
              ll.bitcode_string(ll.current_function()));

    if (ll.debug_is_enabled())
        ll.debug_pop_function();

    ll.end_builder();  // clear the builder

    return ll.current_function();
}

// OptiX Callables:
//  Builds three OptiX callables: an init wrapper, an entry layer wrapper,
//  and a "fused" callable that wraps both and owns the groupdata params buffer.
//
//  Clients can either call both init + entry, or use the fused callable.
//
//  The init and entry layer wrappers exist instead of keeping the underlying
//  functions as direct callables because the fused callable can't call other
//  direct callables.
//
std::vector<llvm::Function*>
BackendLLVM::build_llvm_optix_callables()
{
    std::vector<llvm::Function*> funcs;

    // Build a callable for the entry layer function
    {
        int nlayers               = group().nlayers();
        ShaderInstance* inst      = group()[nlayers - 1];
        std::string dc_entry_name = layer_function_name(group(), *inst, true);

        ll.current_function(
            ll.make_function(dc_entry_name, false,
                             ll.type_void(),  // return type
                             {
                                 llvm_type_sg_ptr(), llvm_type_groupdata_ptr(),
                                 ll.type_void_ptr(),  // userdata_base_ptr
                                 ll.type_void_ptr(),  // output_base_ptr
                                 ll.type_int(),
                                 ll.type_void_ptr(),  // interactive params
                             }));

        llvm::BasicBlock* entry_bb = ll.new_basic_block(dc_entry_name);
        ll.new_builder(entry_bb);

        llvm::Value* args[] = {
            ll.current_function_arg(0), ll.current_function_arg(1),
            ll.current_function_arg(2), ll.current_function_arg(3),
            ll.current_function_arg(4), ll.current_function_arg(5),
        };

        // Call layer
        std::string layer_name = layer_function_name(group(), *inst);
        ll.call_function(layer_name.c_str(), args);

        ll.op_return();
        ll.end_builder();

        funcs.push_back(ll.current_function());
    }

    // Build a callable for the init function
    {
        std::string dc_init_name = init_function_name(shadingsys(), group(),
                                                      true);

        ll.current_function(
            ll.make_function(dc_init_name, false,
                             ll.type_void(),  // return type
                             {
                                 llvm_type_sg_ptr(), llvm_type_groupdata_ptr(),
                                 ll.type_void_ptr(),  // userdata_base_ptr
                                 ll.type_void_ptr(),  // output_base_ptr
                                 ll.type_int(),
                                 ll.type_void_ptr(),  // interactive params
                             }));

        llvm::BasicBlock* init_bb = ll.new_basic_block(dc_init_name);
        ll.new_builder(init_bb);

        llvm::Value* args[] = {
            ll.current_function_arg(0), ll.current_function_arg(1),
            ll.current_function_arg(2), ll.current_function_arg(3),
            ll.current_function_arg(4), ll.current_function_arg(5),
        };

        // Call init
        std::string init_name = init_function_name(shadingsys(), group());
        ll.call_function(init_name.c_str(), args);

        ll.op_return();
        ll.end_builder();

        funcs.push_back(ll.current_function());
    }

    funcs.push_back(build_llvm_fused_callable());
    return funcs;
}

//
// Fused callable:
//  Alternative OptiX API to the init + entry callables.
//
//  Calls init and the entry layer functions itself, so that OSL can own
//  the groupdata params buffer.
//
//  With max_optix_groupdata_alloc > 0, the callable will try to allocate
//  a buffer for groupdata params on the stack. If the buffer requirement
//  exceeds max_optix_groupdata_alloc, it will skip allocation and instead
//  forward the pointer passed in by the renderer.
//
llvm::Function*
BackendLLVM::build_llvm_fused_callable(void)
{
    std::string fused_name = fused_function_name(group());

    // Start building the fused function
    ll.current_function(
        ll.make_function(fused_name, false,
                         ll.type_void(),  // return type
                         {
                             llvm_type_sg_ptr(), llvm_type_groupdata_ptr(),
                             ll.type_void_ptr(),  // userdata_base_ptr
                             ll.type_void_ptr(),  // output_base_ptr
                             ll.type_int(),
                             ll.type_void_ptr(),  // interactive params
                         }));

    llvm::BasicBlock* entry_bb = ll.new_basic_block(fused_name);
    ll.new_builder(entry_bb);

    // If it fits, allocate a groupdata params buffer and overwrite the
    // renderer-supplied pointer
    llvm::Value* llvm_groupdata_ptr = ll.current_function_arg(1);

    if ((int)group().llvm_groupdata_size()
        <= shadingsys().m_max_optix_groupdata_alloc)
        llvm_groupdata_ptr = ll.op_alloca(m_llvm_type_groupdata, 1,
                                          "groupdata_buffer", 8);

    llvm::Value* args[] = {
        ll.current_function_arg(0), llvm_groupdata_ptr,
        ll.current_function_arg(2), ll.current_function_arg(3),
        ll.current_function_arg(4), ll.current_function_arg(5),
    };

    // Call init
    std::string init_name = init_function_name(shadingsys(), group());
    ll.call_function(init_name.c_str(), args);

    int nlayers          = group().nlayers();
    ShaderInstance* inst = group()[nlayers - 1];

    // Call entry
    std::string layer_name = layer_function_name(group(), *inst);
    ll.call_function(layer_name.c_str(), args);

    ll.op_return();
    ll.end_builder();

    return ll.current_function();
}

llvm::Function*
BackendLLVM::build_llvm_instance(bool groupentry)
{
    // Make a layer function: void layer_func(ShaderGlobals*, GroupData*)
    // Note that the GroupData* is passed as a void*.
    std::string unique_layer_name = layer_function_name(group(), *inst());

    bool is_entry_layer = group().is_entry_layer(layer());
    ll.current_function(ll.make_function(
        unique_layer_name,
        !is_entry_layer,  // fastcall for non-entry layer functions
        ll.type_void(),   // return type
        {
            llvm_type_sg_ptr(), llvm_type_groupdata_ptr(),
            ll.type_void_ptr(),  // userdata_base_ptr
            ll.type_void_ptr(),  // output_base_ptr
            ll.type_int(),
            ll.type_void_ptr(),  // FIXME: interactive_params
        }));

    if (ll.debug_is_enabled()) {
        const Opcode& mainbegin(inst()->op(inst()->maincodebegin()));
        ll.debug_push_function(unique_layer_name, mainbegin.sourcefile(),
                               mainbegin.sourceline());
    }

    // Get shader globals and groupdata pointers
    m_llvm_shaderglobals_ptr = ll.current_function_arg(0);  //arg_it++;
    m_llvm_shaderglobals_ptr->setName("shaderglobals_ptr");
    m_llvm_groupdata_ptr = ll.current_function_arg(1);  //arg_it++;
    m_llvm_groupdata_ptr->setName("groupdata_ptr");
    m_llvm_userdata_base_ptr = ll.current_function_arg(2);  //arg_it++;
    m_llvm_userdata_base_ptr->setName("userdata_base_ptr");
    m_llvm_output_base_ptr = ll.current_function_arg(3);  //arg_it++;
    m_llvm_output_base_ptr->setName("output_base_ptr");
    m_llvm_shadeindex = ll.current_function_arg(4);  //arg_it++;
    m_llvm_shadeindex->setName("shadeindex");
    m_llvm_interactive_params_ptr = ll.current_function_arg(5);  //arg_it++;
    m_llvm_interactive_params_ptr->setName("interactive_params_ptr");

    llvm::BasicBlock* entry_bb = ll.new_basic_block(unique_layer_name);
    m_exit_instance_block      = NULL;

    // Set up a new IR builder
    ll.new_builder(entry_bb);

    llvm::Value* layerfield = layer_run_ref(layer_remap(layer()));
    if (is_entry_layer && !group().is_last_layer(layer())) {
        // For entry layers, we need an extra check to see if it already
        // ran. If it has, do an early return. Otherwise, set the 'ran' flag
        // and then run the layer.
        if (shadingsys().llvm_debug_layers())
            llvm_gen_debug_printf(
                fmtformat("checking for already-run layer {} {} {}",
                          this->layer(), inst()->layername(),
                          inst()->shadername()));
        llvm::Value* executed = ll.op_eq(ll.op_load(ll.type_bool(), layerfield),
                                         ll.constant_bool(true));
        llvm::BasicBlock* then_block  = ll.new_basic_block();
        llvm::BasicBlock* after_block = ll.new_basic_block();
        ll.op_branch(executed, then_block, after_block);
        // insert point is now then_block
        // we've already executed, so return early
        if (shadingsys().llvm_debug_layers())
            llvm_gen_debug_printf(fmtformat(
                "  taking early exit, already executed layer {} {} {}",
                this->layer(), inst()->layername(), inst()->shadername()));
        ll.op_return();
        ll.set_insert_point(after_block);
    }

    if (shadingsys().llvm_debug_layers())
        llvm_gen_debug_printf(fmtformat("enter layer {} {} {}", this->layer(),
                                        inst()->layername(),
                                        inst()->shadername()));
    // Mark this layer as executed
    if (!group().is_last_layer(layer())) {
        ll.op_store(ll.constant_bool(true), layerfield);
        if (shadingsys().countlayerexecs())
            ll.call_function("osl_incr_layers_executed", sg_void_ptr());
    }

    // Setup the symbols
    m_named_values.clear();
    m_layers_already_run.clear();
    for (auto&& s : inst()->symbols()) {
        // Skip constants -- we always inline scalar constants, and for
        // array constants we will just use the pointers to the copy of
        // the constant that belongs to the instance.
        if (s.symtype() == SymTypeConst)
            continue;
        // Skip structure placeholders
        if (s.typespec().is_structure())
            continue;
        // Allocate space for locals, temps, aggregate constants, and some output params
        if (s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp
            || s.symtype() == SymTypeConst || can_treat_param_as_local(s))
            getOrAllocateLLVMSymbol(s);
        // Set initial value for constants, closures, and strings that are
        // not parameters.
        if (s.symtype() != SymTypeParam && s.symtype() != SymTypeOutputParam
            && s.symtype() != SymTypeGlobal
            && (s.is_constant() || s.typespec().is_closure_based()
                || s.typespec().is_string_based()
                || ((s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp)
                    && shadingsys().debug_uninit())))
            llvm_assign_initial_value(s);
        // If debugnan is turned on, globals check that their values are ok
        if (s.symtype() == SymTypeGlobal && shadingsys().debug_nan()) {
            TypeDesc t = s.typespec().simpletype();
            if (t.basetype
                == TypeDesc::FLOAT) {  // just check float-based types
                int ncomps = t.numelements() * t.aggregate;
                llvm::Value* args[]
                    = { ll.constant(ncomps),
                        llvm_void_ptr(s),
                        ll.constant((int)s.has_derivs()),
                        sg_void_ptr(),
                        llvm_load_stringhash(inst()->shadername()),
                        ll.constant(0),
                        llvm_load_stringhash(s.unmangled()),
                        ll.constant(0),
                        ll.constant(ncomps),
                        llvm_load_stringhash("<none>") };
                ll.call_function("osl_naninf_check", args);
            }
        }
    }
    // make a second pass for the parameters (which may make use of
    // locals and constants from the first pass)
    FOREACH_PARAM(Symbol & s, inst())
    {
        // Skip structure placeholders
        if (s.typespec().is_structure())
            continue;
        // Skip if it's never read and isn't connected
        if (!s.everread() && !s.connected_down() && !s.connected()
            && !s.renderer_output())
            continue;
        // Skip if it's an interpolated (userdata) parameter and we're
        // initializing them lazily, or if it's an interactively-adjusted
        // parameter.
        if ((s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam)
            && !s.typespec().is_closure() && !s.connected()
            && !s.connected_down()
            && (s.interactive()
                || (s.interpolated() && shadingsys().lazy_userdata())))
            continue;
        // Set initial value for params (may contain init ops)
        llvm_assign_initial_value(s);
    }

    // All the symbols are stack allocated now.

    if (groupentry) {
        // Group entries also need to run any earlier layers that must be
        // run unconditionally. It's important that we do this AFTER all the
        // parameter initialization for this layer.
        for (int i = 0; i < group().nlayers() - 1; ++i) {
            ShaderInstance* gi = group()[i];
            if (!gi->unused() && !gi->empty_instance() && !gi->run_lazily())
                llvm_call_layer(i, true /* unconditionally run */);
        }
    }

    // Mark all the basic blocks, including allocating llvm::BasicBlock
    // records for each.
    find_basic_blocks();
    find_conditionals();
    m_call_layers_inserted.clear();

    build_llvm_code(inst()->maincodebegin(), inst()->maincodeend());

    if (llvm_has_exit_instance_block())
        ll.op_branch(m_exit_instance_block);  // also sets insert point

    // Track all symbols who needed 'partial' initialization
    std::unordered_set<Symbol*> initedsyms;

    // Transfer all of this layer's outputs into the downstream shader's
    // inputs.
    for (int layer = this->layer() + 1; layer < group().nlayers(); ++layer) {
        // If the connection is to a layer known to not be used, the copy
        // can be skipped.
        if (m_layer_remap[layer] == -1)
            continue;
        ShaderInstance* child = group()[layer];
        for (int c = 0, Nc = child->nconnections(); c < Nc; ++c) {
            const Connection& con(child->connection(c));
            if (con.srclayer == this->layer()) {
                OSL_ASSERT(
                    con.src.arrayindex == -1 && con.dst.arrayindex == -1
                    && "no support for individual array element connections");
                // Validate unsupported connection vecSrc -> vecDst[j]
                OSL_ASSERT((con.dst.channel == -1
                            || con.src.type.aggregate() == TypeDesc::SCALAR
                            || con.src.channel != -1)
                           && "no support for vector -> vector[i] connections");

                Symbol* srcsym(inst()->symbol(con.src.param));
                Symbol* dstsym(child->symbol(con.dst.param));

                // Check remaining connections to see if any channels of this
                // aggregate need to be initialize.
                if (con.dst.channel != -1 && initedsyms.count(dstsym) == 0) {
                    initedsyms.insert(dstsym);
                    std::bitset<32> inited(0);  // Only need to be 16 (matrix4)
                    OSL_DASSERT(dstsym->typespec().aggregate()
                                <= inited.size());
                    unsigned ninit = dstsym->typespec().aggregate() - 1;
                    for (int rc = c + 1; rc < Nc && ninit; ++rc) {
                        const Connection& next(child->connection(rc));
                        if (next.srclayer == this->layer()) {
                            // Allow redundant/overwriting connections, i.e:
                            // 1.  connect layer.value[i] connect layer.value[j]
                            // 2.  connect layer.value connect layer.value
                            if (child->symbol(next.dst.param) == dstsym) {
                                if (next.dst.channel != -1) {
                                    OSL_DASSERT(next.dst.channel
                                                < (int)inited.size());
                                    if (!inited[next.dst.channel]) {
                                        inited[next.dst.channel] = true;
                                        --ninit;
                                    }
                                } else
                                    ninit = 0;
                            }
                        }
                    }
                    if (ninit) {
                        // FIXME: Init only components that are not connected
                        llvm_assign_initial_value(*dstsym, true);
                    }
                }

                // llvm_run_connected_layers tracks layers that have been run,
                // so no need to do it here as well
                llvm_run_connected_layers(*srcsym, con.src.param);

                // FIXME -- I'm not sure I understand this.  Isn't this
                // unnecessary if we wrote to the parameter ourself?
                llvm_assign_impl(*dstsym, *srcsym, -1, con.src.channel,
                                 con.dst.channel);
            }
        }
    }
    // llvm_gen_debug_printf ("done copying connections");

    // Copy results to renderer outputs
    llvm::Value* sindex = nullptr;
    FOREACH_PARAM(Symbol & s, inst())
    {
        if (!s.renderer_output())  // Skip if not a renderer output
            continue;
        // Try to look up the sym among the outputs with the full layer.name
        // specification first. If that fails, look for name only.
        ustring layersym = ustring::fmtformat("{}.{}", inst()->layername(),
                                              s.name());
        auto symloc      = group().find_symloc(layersym, SymArena::Outputs);
        if (!symloc)
            symloc = group().find_symloc(s.name(), SymArena::Outputs);
        if (!symloc) {
            // std::cout << "No output copy for " << s.name()
            //           << " because no symloc was found\n";
            continue;  // not found in either place
        }

        if (!equivalent(s.typespec(), symloc->type)
            || s.typespec().is_closure()) {
            std::cout << "No output copy for " << s.typespec() << ' '
                      << s.name()
                      << " because of type mismatch vs symloc=" << symloc->type
                      << "\n";
            continue;  // types didn't match
        }

        int size = int(symloc->type.size());
        if (symloc->derivs && s.has_derivs())
            size *= 3;  // If we're copying the derivs

        // std::cout << "GEN found output " << s.name() << " -> "
        //           << symloc->name << ' ' << s.typespec() << " size="
        //           << symloc->type.size() << "\n";
        llvm::Value* srcptr = llvm_void_ptr(s);
        // llvm::Value* offset = ll.constanti64(symloc->offset);
        // llvm::Value* stride = ll.constanti64(symloc->stride);
        if (!sindex)
            sindex = ll.op_int_to_longlong(m_llvm_shadeindex);
        // llvm::Value* fulloffset = ll.op_add(offset, ll.op_mul(stride, sindex));
        // llvm::Value* dstptr = ll.offset_ptr(m_llvm_output_base_ptr, fulloffset);
        llvm::Value* dstptr = symloc_ptr(symloc, m_llvm_output_base_ptr,
                                         sindex);
        ll.op_memcpy(dstptr, srcptr, size);
        // Clear derivs if output wants derivs but source didn't have them
        if (symloc->derivs && !s.has_derivs())
            ll.op_memset(ll.offset_ptr(dstptr, size), 0, 2 * size);
    }

    // All done
    if (shadingsys().llvm_debug_layers())
        llvm_gen_debug_printf(fmtformat("exit layer {} {} {}", this->layer(),
                                        inst()->layername(),
                                        inst()->shadername()));
    ll.op_return();

    if (llvm_debug())
        std::cout << "layer_func (" << unique_layer_name << ") "
                  << this->layer() << "/" << group().nlayers()
                  << " after llvm  = "
                  << ll.bitcode_string(ll.current_function()) << "\n";

    if (ll.debug_is_enabled()) {
        ll.debug_pop_function();
    }

    ll.end_builder();  // clear the builder

    return ll.current_function();
}



void
BackendLLVM::initialize_llvm_group()
{
    if (ll.debug_is_enabled()) {
        const char* compile_unit_name = m_group.m_name.empty()
                                            ? unknown_shader_group_name.c_str()
                                            : m_group.m_name.c_str();
        ll.debug_setup_compilation_unit(compile_unit_name);
    }

    // Set up optimization passes. Don't target the host if we're building
    // for OptiX.
    ll.setup_optimization_passes(shadingsys().llvm_optimize(),
                                 shadingsys().llvm_target_host()
                                     && !use_optix());

    // Clear the shaderglobals and groupdata types -- they will be
    // created on demand.
    m_llvm_type_sg                = NULL;
    m_llvm_type_groupdata         = NULL;
    m_llvm_type_closure_component = NULL;

    initialize_llvm_helper_function_map();

    // Skipping this in the non-JIT OptiX case suppresses an LLVM warning
    if (!use_optix())
        ll.InstallLazyFunctionCreator(helper_function_lookup);

    for (HelperFuncMap::iterator i = llvm_helper_function_map.begin(),
                                 e = llvm_helper_function_map.end();
         i != e; ++i) {
        const std::string& funcname(i->first);
        bool varargs      = false;
        const char* types = i->second.argtypes;
        int advance;
        TypeSpec rettype = OSLCompilerImpl::type_from_code(types, &advance);
        types += advance;
        std::vector<llvm::Type*> params;
        while (*types) {
            TypeSpec t = OSLCompilerImpl::type_from_code(types, &advance);
            if (t.simpletype().basetype == TypeDesc::UNKNOWN) {
                OSL_DASSERT(*types == '*');
                if (*types == '*')
                    varargs = true;
            } else {
                params.push_back(llvm_pass_type(t));
            }
            types += advance;
        }
#if OSL_USE_OPTIX
        if (varargs && use_optix()) {
            varargs = false;
            params.push_back(ll.type_void_ptr());
        }
#endif
        llvm::Function* f = ll.make_function(funcname, false,
                                             llvm_type(rettype), params,
                                             varargs);

        // Skipping this in the non-JIT OptiX case suppresses an LLVM warning
        if (!use_optix())
            ll.add_function_mapping(f, (void*)i->second.function);
    }

    // Needed for closure setup
    std::vector<llvm::Type*> params(3);
    params[0]                        = (llvm::Type*)ll.type_char_ptr();
    params[1]                        = ll.type_int();
    params[2]                        = (llvm::Type*)ll.type_char_ptr();
    m_llvm_type_prepare_closure_func = ll.type_function_ptr(ll.type_void(),
                                                            params);
    m_llvm_type_setup_closure_func   = m_llvm_type_prepare_closure_func;
}


void
BackendLLVM::prepare_module_for_cuda_jit()
{
    auto is_inline_fn = [&](const std::string& name) {
        return shadingsys().m_inline_functions.find(ustring(name))
               != shadingsys().m_inline_functions.end();
    };

    auto is_noinline_fn = [&](const std::string& name) {
        return shadingsys().m_noinline_functions.find(ustring(name))
               != shadingsys().m_noinline_functions.end();
    };

    const bool no_inline = shadingsys().optix_no_inline();
    const bool no_inline_layer_funcs
        = shadingsys().optix_no_inline_layer_funcs();
    const bool merge_layer_funcs  = shadingsys().optix_merge_layer_funcs();
    const bool no_inline_rend_lib = shadingsys().optix_no_inline_rend_lib();
    const int no_inline_thresh    = shadingsys().optix_no_inline_thresh();
    const int force_inline_thresh = shadingsys().optix_force_inline_thresh();

    // Adjust the linkage for the library and group functions:
    //  * Set external linkage for the library functions to prevent the
    //    function signatures from being changed by dead arg elimination
    //    passes. The signatures need to match the shadeops PTX module.
    //
    //  * Set private linkage for the layer functions to help avoid
    //    collisions between layer functions from different ShaderGroups.
    for (llvm::Function& fn : *ll.module()) {
        if (fn.hasFnAttribute("osl-lib-function")) {
            fn.setLinkage(llvm::GlobalValue::ExternalLinkage);
        }
#if OSL_LLVM_VERSION >= 180
        else if (fn.getName().starts_with(group().name().c_str()))
#else
        else if (fn.getName().startswith(group().name().c_str()))
#endif
        {
            fn.setLinkage(llvm::GlobalValue::PrivateLinkage);
        }
    }

    // Set the inlining behavior for each function in the module, based on
    // the shadingsys attributes. The inlining attributes are not modified
    // by default.
    for (llvm::Function& fn : *ll.module()) {
        // Don't modify the inlining attribute for:
        //  * group entry functions
        //  * llvm library functions
#if OSL_LLVM_VERSION >= 180
        if (fn.getName().starts_with("__direct_callable__")
            || fn.getName().starts_with("llvm."))
            continue;
#else
        if (fn.getName().startswith("__direct_callable__")
            || fn.getName().startswith("llvm."))
            continue;
#endif

        // Merge layer functions which are only called from one place
        if (merge_layer_funcs && !fn.hasFnAttribute("osl-lib-function")
            && fn.hasOneUse()) {
            fn.addFnAttr(llvm::Attribute::AlwaysInline);
            continue;
        }

        // Inline the functions registered with the ShadingSystem
        if (is_inline_fn(fn.getName().str())) {
            fn.addFnAttr(llvm::Attribute::AlwaysInline);
            continue;
        }

        // No-inline the functions registered with the ShadingSystem
        if (is_noinline_fn(fn.getName().str())) {
            fn.deleteBody();
            continue;
        }

        if (no_inline) {
            fn.addFnAttr(llvm::Attribute::NoInline);

            // Delete the bodies of library functions which will never be inlined.
            // This reduces the size of the module prior to opt/JIT.
            if (fn.hasFnAttribute("osl-lib-function")) {
                fn.deleteBody();
            }
            continue;
        }

        if (no_inline_rend_lib && fn.hasFnAttribute("osl-rend_lib-function")) {
            fn.deleteBody();
            continue;
        }

        if (no_inline_layer_funcs && !fn.hasFnAttribute("osl-lib-function")) {
            fn.addFnAttr(llvm::Attribute::NoInline);
            continue;
        }

        // Only apply the inline thresholds to library functions.
        if (!fn.hasFnAttribute("osl-lib-function")) {
            continue;
        }

        const int inst_count = fn.getInstructionCount();
        if (inst_count >= no_inline_thresh) {
            fn.deleteBody();
        } else if (inst_count > 0 && inst_count <= force_inline_thresh) {
            fn.addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }

#ifndef OSL_CUDA_NO_FTZ
    for (llvm::Function& fn : *ll.module()) {
        fn.addFnAttr("nvptx-f32ftz", "true");
        fn.addFnAttr("denormal-fp-math", "preserve-sign,preserve-sign");
        fn.addFnAttr("denormal-fp-math-f32", "preserve-sign,preserve-sign");
    }
#endif
}



static void
empty_group_func(void*, void*)
{
}



void
BackendLLVM::run()
{
    if (group().does_nothing()) {
        group().llvm_compiled_init((RunLLVMGroupFunc)empty_group_func);
        group().llvm_compiled_version((RunLLVMGroupFunc)empty_group_func);
        return;
    }

    // At this point, we already hold the lock for this group, by virtue
    // of ShadingSystemImpl::optimize_group.
    OIIO::Timer timer;
    std::string err;

    {
#ifdef OSL_LLVM_NO_BITCODE
        // I don't know which exact part has thread safety issues, but it
        // crashes on windows when we don't lock.
        // FIXME -- try subsequent LLVM releases on Windows to see if this
        // is a problem that is eventually fixed on the LLVM side.
        static spin_mutex mutex;
        OIIO::spin_lock lock(mutex);
#endif

#ifdef OSL_LLVM_NO_BITCODE
        OSL_ASSERT(!use_rs_bitcode());
        ll.module(ll.new_module("llvm_ops"));
#    if OSL_USE_OPTIX
        if (use_optix()) {
            // If the module is created from LLVM bitcode, the target and
            // data layout is inherited from that, but if creating an empty
            // module like here, have to manually set those, otherwise
            // compiling will later fail because the NVPTX target is not found.
            // The target triple and data layout used here are those specified
            // for NVPTX (https://www.llvm.org/docs/NVPTXUsage.html#triples).
            ll.module()->setDataLayout(
                "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            ll.module()->setTargetTriple("nvptx64-nvidia-cuda");
        }
#    endif
#else
        if (!use_optix()) {
            if (use_rs_bitcode()) {
                ll.module(ll.module_from_bitcode(
                    (char*)osl_llvm_compiled_rs_dependent_ops_block,
                    osl_llvm_compiled_rs_dependent_ops_size,
                    "llvm_rs_dependent_ops", &err));
                if (err.length())
                    shadingcontext()->errorfmt(
                        "llvm::parseBitcodeFile returned '{}' for llvm_rs_dependent_ops\n",
                        err);

//Leaving this around for developers to make sure LLVM's shaderglobals and C++'s are binary compatible
#    if 0
                std::vector<unsigned int> offset_by_index;
                build_offsets_of_ShaderGlobals(offset_by_index);
                ll.validate_struct_data_layout(m_llvm_type_sg, offset_by_index);
#    endif

                std::vector<char>& rs_free_function_bitcode
                    = shadingsys().m_rs_bitcode;
                OSL_ASSERT(rs_free_function_bitcode.size()
                           && "Free Function bitcode is empty");

                llvm::Module* rs_free_functions_module = ll.module_from_bitcode(
                    static_cast<const char*>(rs_free_function_bitcode.data()),
                    rs_free_function_bitcode.size(), "rs_free_functions", &err);
                if (err.length())
                    shadingcontext()->errorfmt(
                        "llvm::parseBitcodeFile returned '{}' for rs_free_functions\n",
                        err);
                std::unique_ptr<llvm::Module> rs_free_functions_module_ptr(
                    rs_free_functions_module);
                bool success = ll.absorb_module(
                    std::move(rs_free_functions_module_ptr));
                if (!success)
                    shadingcontext()->errorfmt(
                        "LLVM_Util::absorb_module failed'\n");
            } else {
                ll.module(
                    ll.module_from_bitcode((char*)osl_llvm_compiled_ops_block,
                                           osl_llvm_compiled_ops_size,
                                           "llvm_ops", &err));
                if (err.length())
                    shadingcontext()->errorfmt(
                        "llvm::parseBitcodeFile returned '{}' for llvm_ops\n",
                        err);
            }

        } else {
#    ifdef OSL_LLVM_CUDA_BITCODE
            llvm::Module* shadeops_module = ll.module_from_bitcode(
                (char*)shadeops_cuda_llvm_compiled_ops_block,
                shadeops_cuda_llvm_compiled_ops_size, "llvm_ops", &err);

            if (err.length())
                shadingcontext()->errorfmt(
                    "llvm::parseBitcodeFile returned '{}' for cuda llvm_ops\n",
                    err);

            shadeops_module->setDataLayout(
                "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            shadeops_module->setTargetTriple("nvptx64-nvidia-cuda");

            std::unique_ptr<llvm::Module> shadeops_ptr(shadeops_module);
            llvm::Linker::linkModules(*ll.module(), std::move(shadeops_ptr),
                                      llvm::Linker::Flags::None);

            if (err.length())
                shadingcontext()->errorfmt(
                    "llvm::parseBitcodeFile returned '{}' for cuda rend_lib\n",
                    err);

            // The renderer may provide additional shadeops bitcode for renderer-specific
            // functionality ("rend_lib" fuctions). Like the built-in shadeops, the rend_lib
            // functions may or may not be inlined, depending on the optimization options.
            std::vector<char>& bitcode = shadingsys().m_lib_bitcode;
            if (bitcode.size()) {
                llvm::Module* rend_lib_module = ll.module_from_bitcode(
                    static_cast<const char*>(bitcode.data()), bitcode.size(),
                    "cuda_rend_lib", &err);

                if (err.length())
                    shadingcontext()->errorfmt(
                        "llvm::parseBitcodeFile returned '{}' for cuda llvm_ops\n",
                        err);

                rend_lib_module->setDataLayout(
                    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
                rend_lib_module->setTargetTriple("nvptx64-nvidia-cuda");

                for (llvm::Function& fn : *rend_lib_module) {
                    fn.addFnAttr("osl-rend_lib-function", "true");
                }

                std::unique_ptr<llvm::Module> rend_lib_ptr(rend_lib_module);
                llvm::Linker::linkModules(*ll.module(), std::move(rend_lib_ptr),
                                          llvm::Linker::Flags::OverrideFromSrc);
            }
#    else
            OSL_ASSERT(0 && "Must generate LLVM CUDA bitcode for OptiX");
#    endif
            // Ensure that the correct target triple and data layout are set when targeting NVPTX.
            // The triple is empty with recent versions of LLVM (e.g., 15) for reasons that aren't
            // clear. So we must set them to the expected values.
            // See: https://llvm.org/docs/NVPTXUsage.html
            ll.module()->setTargetTriple("nvptx64-nvidia-cuda");
            ll.module()->setDataLayout(
                "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");

            // Tag each function as an OSL library function to help with
            // inlining and optimization after codegen.
            for (llvm::Function& fn : *ll.module()) {
                fn.addFnAttr("osl-lib-function", "true");
            }

            // Mark all global variables extern and discard their initializers.
            // Global variables are defined in the shadeops PTX file.
            for (llvm::GlobalVariable& global : ll.module()->globals()) {
                global.setLinkage(llvm::GlobalValue::ExternalLinkage);
                global.setExternallyInitialized(true);
                global.setInitializer(nullptr);
                // Replace characters not supported in ptx, matching the LLVM
                // NVPTXAssignValidGlobalNames pass.
                string_view global_name(global.getName().data(),
                                        global.getName().size());
                if (Strutil::contains_any_char(global_name, ".@")) {
                    std::string valid_name = global_name;
                    valid_name = Strutil::replace(valid_name, ".", "_$_", true);
                    valid_name = Strutil::replace(valid_name, "@", "_$_", true);
                    global.setName(valid_name);
                }
            }
        }
        OSL_ASSERT(ll.module());
#endif

        // Create the ExecutionEngine. We don't create an ExecutionEngine in the
        // OptiX case, because we are using the NVPTX backend and not MCJIT. However,
        // it's still useful to set the target ISA to facilitate PTX-specific codegen.
        if (use_optix()) {
            ll.set_target_isa(TargetISA::NVPTX);
        } else if (!ll.make_jit_execengine(
                       &err,
                       ll.lookup_isa_by_name(shadingsys().m_llvm_jit_target),
                       shadingsys().llvm_debugging_symbols(),
                       shadingsys().llvm_profiling_events())) {
            shadingcontext()->errorfmt("Failed to create engine: {}\n", err);
            OSL_ASSERT(0);
            return;
        }

        // End of mutex lock, for the OSL_LLVM_NO_BITCODE case
    }

    m_stat_llvm_setup_time += timer.lap();

    // Set up m_num_used_layers to be the number of layers that are
    // actually used, and m_layer_remap[] to map original layer numbers
    // to the shorter list of actually-called layers. We also note that
    // if m_layer_remap[i] is < 0, it's not a layer that's used.
    int nlayers = group().nlayers();
    m_layer_remap.resize(nlayers, -1);
    m_num_used_layers = 0;
    if (debug() >= 1)
        std::cout << "\nLayers used: (group " << group().name() << ")\n";
    for (int layer = 0; layer < nlayers; ++layer) {
        // Skip unused or empty layers, unless they are callable entry
        // points.
        ShaderInstance* inst = group()[layer];
        bool is_single_entry = (layer == (nlayers - 1)
                                && group().num_entry_layers() == 0);
        if (inst->entry_layer() || is_single_entry
            || (!inst->unused() && !inst->empty_instance())) {
            if (debug() >= 1)
                std::cout << "  " << layer << ' ' << inst->layername() << "\n";
            m_layer_remap[layer] = m_num_used_layers++;
        }
    }
    shadingsys().m_stat_empty_instances += nlayers - m_num_used_layers;

    initialize_llvm_group();

    // Generate the LLVM IR for each layer.  Skip unused layers.
    m_llvm_local_mem          = 0;
    llvm::Function* init_func = build_llvm_init();
    std::vector<llvm::Function*> funcs(nlayers, NULL);
    for (int layer = 0; layer < nlayers; ++layer) {
        set_inst(layer);
        if (m_layer_remap[layer] != -1) {
            // If no entry points were specified, the last layer is special,
            // it's the single entry point for the whole group.
            bool is_single_entry = (layer == (nlayers - 1)
                                    && group().num_entry_layers() == 0);
            funcs[layer]         = build_llvm_instance(is_single_entry);
        }
    }

    std::vector<llvm::Function*> optix_externals;
    if (use_optix())
        optix_externals = build_llvm_optix_callables();

    // llvm::Function* entry_func = group().num_entry_layers() ? NULL : funcs[m_num_used_layers-1];
    m_stat_llvm_irgen_time += timer.lap();

    if (shadingsys().m_max_local_mem_KB
        && m_llvm_local_mem / 1024 > shadingsys().m_max_local_mem_KB) {
        shadingcontext()->errorfmt(
            "Shader group \"{}\" needs too much local storage: {} KB",
            group().name(), m_llvm_local_mem / 1024);
    }

    // The module contains tons of "library" functions that our generated IR
    // might call. But probably not. We don't want to incur the overhead of
    // fully compiling those, so we want to get rid of all functions not
    // called directly or indirectly by our init or shader layer functions.
    // It turns out to be much faster to prune these from the IR ahead of
    // time versus letting the optimizer figure it out as the optimizer
    // would have run many passes over functions which we later will be
    // dropped. Only the external_functions will have external linkage all
    // other remaining functions will be set to internal linkage.
    if (shadingsys().llvm_prune_ir_strategy() == "none") {
        // Do nothing! This is only useful for testing how much we reduce
        // optimization and JIT time with the other strategies.
    } else /* if (shadingsys().llvm_prune_ir_strategy() == "prune") */ {
        // New (2020) and default behavior, from Alex Wells:
        // Full prune of uncalled functions, and marking as 'internal' to
        // the module all but our known entry points from the outside. This
        // seems to yield about another 5-10% opt+JIT speed gain versus
        // merely internalizing.
        std::unordered_set<llvm::Function*> external_functions;
        if (use_optix()) {
            for (llvm::Function* func : optix_externals)
                external_functions.insert(func);
        } else {
            external_functions.insert(init_func);

            for (int layer = 0; layer < nlayers; ++layer) {
                llvm::Function* f = funcs[layer];
                // If we plan to call bitcode_string of a layer's function after
                // optimization it may not exist after optimization unless we
                // treat it as external.
                if (f && (group().is_entry_layer(layer) || llvm_debug())) {
                    external_functions.insert(f);
                }
            }
        }
        ll.prune_and_internalize_module(external_functions);
    }

    // Debug code to dump the pre-optimized bitcode to a file
    if (llvm_debug() >= 2 || shadingsys().llvm_output_bitcode()) {
        // Make a safe group name that doesn't have "/" in it! Also beware
        // filename length limits.
        std::string safegroup;
        safegroup = Strutil::replace(group().name(), "/", "_", true);
        safegroup = Strutil::replace(safegroup, ":", "_", true);
        if (safegroup.size() > 235)
            safegroup = fmtformat("TRUNC_{}_{}",
                                  safegroup.substr(safegroup.size() - 235),
                                  group().id());
        std::string name = fmtformat("{}.ll", safegroup);
        OIIO::ofstream out;
        OIIO::Filesystem::open(out, name);
        if (out) {
            out << ll.bitcode_string(ll.module());
            shadingsys().infofmt("Wrote pre-optimized bitcode to '{}'", name);
        } else {
            shadingsys().errorfmt("Could not write to '{}'", name);
        }
    }

    if (use_rs_bitcode()) {
        std::vector<std::string> names_of_unmapped_globals;
        ll.validate_global_mappings(names_of_unmapped_globals);
        if (!names_of_unmapped_globals.empty()) {
            shadingsys().errorfmt(
                "Renderers should call OSL::register_JIT_Global(const char* global_var_name, void* global_var_addr) for each global variable used by its free function renderer services bitcode");
            for (const auto& unmapped_name : names_of_unmapped_globals) {
                shadingsys().errorfmt(
                    ">>>>External global variable {} was not mapped to an address!",
                    unmapped_name);
            }
        }
    }

#if OSL_USE_OPTIX
    if (use_optix()) {
        // Set some extra LLVM Function attributes before optimizing the Module.
        prepare_module_for_cuda_jit();
    }
#endif

    // Optimize the LLVM IR unless it's a do-nothing group.
    if (!group().does_nothing()) {
        ll.do_optimize();
    }

#if OSL_USE_OPTIX
    if (use_optix()) {
        // Drop everything but the init and group entry functions and generated
        // group functions. The definitions for the non-inlined library
        // functions are supplied via a separate shadeops PTX module.
        for (llvm::Function& fn : *ll.module()) {
            if (fn.hasFnAttribute("osl-lib-function")) {
                fn.deleteBody();
            }
        }
    }
#endif

    m_stat_llvm_opt_time += timer.lap();

    if (llvm_debug()) {
#if 1
        // Feel it is more useful to get a dump of the entire optimized module
        // vs. individual layer functions.  Especially now because we have pruned all
        // unused function declarations and functions out of the the module.
        // Big benefit is that the module output can be cut and pasted into
        // https://godbolt.org/ compiler explorer as LLVM IR with a LLC target
        // and -mcpu= options to see what machine code will be generated by
        // different LLC versions and cpu targets
        std::cout << "module after opt  = \n" << ll.module_string() << "\n";
#else
        for (auto&& f : funcs)
            if (f)
                shadingsys().infofmt("func after opt  = {}\n",
                                     ll.bitcode_string(f));
#endif
    }

    // Debug code to dump the post-optimized bitcode to a file
    if (llvm_debug() >= 2 || shadingsys().llvm_output_bitcode()) {
        // Make a safe group name that doesn't have "/" in it! Also beware
        // filename length limits.
        std::string safegroup;
        safegroup = Strutil::replace(group().name(), "/", "_", true);
        safegroup = Strutil::replace(safegroup, ":", "_", true);
        if (safegroup.size() > 235)
            safegroup = fmtformat("TRUNC_{}_{}",
                                  safegroup.substr(safegroup.size() - 235),
                                  group().id());
        std::string name = fmtformat("{}_O{}.ll", safegroup,
                                     shadingsys().llvm_optimize());
        OIIO::ofstream out;
        OIIO::Filesystem::open(out, name);
        if (out) {
            out << ll.bitcode_string(ll.module());
            shadingsys().infofmt("Wrote post-optimized bitcode to '{}'", name);
        } else {
            shadingsys().errorfmt("Could not write to '{}'", name);
        }
    }

#if OSL_USE_OPTIX
    if (use_optix()) {
        ll.ptx_compile_group(nullptr, group().name().string(),
                             group().m_llvm_ptx_compiled_version);
        if (group().m_llvm_ptx_compiled_version.empty()) {
            OSL_ASSERT(0 && "Unable to generate PTX");
        }
    } else
#endif
    {
        // Force the JIT to happen now and retrieve the JITed function pointers
        // for the initialization and all public entry points.
        group().llvm_compiled_init(
            (RunLLVMGroupFunc)ll.getPointerToFunction(init_func));
        for (int layer = 0; layer < nlayers; ++layer) {
            llvm::Function* f = funcs[layer];
            if (f && group().is_entry_layer(layer))
                group().llvm_compiled_layer(
                    layer, (RunLLVMGroupFunc)ll.getPointerToFunction(f));
        }
        if (group().num_entry_layers())
            group().llvm_compiled_version(NULL);
        else
            group().llvm_compiled_version(
                group().llvm_compiled_layer(nlayers - 1));
    }

    // We are destroying the entire module below,
    // no reason to bother destroying individual functions
#if 0
    // Remove the IR for the group layer functions, we've already JITed it
    // and will never need the IR again.  This saves memory, and also saves
    // a huge amount of time since we won't re-optimize it again and again
    // if we keep adding new shader groups to the same Module.
    for (int i = 0; i < nlayers; ++i) {
        if (funcs[i])
            ll.delete_func_body (funcs[i]);
    }
    ll.delete_func_body (init_func);
#endif

    // Free the exec and module to reclaim all the memory.  This definitely
    // saves memory, and has almost no effect on runtime.
    ll.execengine(NULL);

    // N.B. Destroying the EE should have destroyed the module as well.
    ll.module(NULL);

    m_stat_llvm_jit_time += timer.lap();

    m_stat_total_llvm_time = timer();

    if (shadingsys().m_compile_report) {
        shadingcontext()->infofmt("JITed shader group {}:", group().name());
        shadingcontext()->infofmt(
            "    ({:1.2f}s = {:1.2f} setup, {:1.2f} ir, {:1.2f} opt, {:1.2f} jit; local mem {}KB)",
            m_stat_total_llvm_time, m_stat_llvm_setup_time,
            m_stat_llvm_irgen_time, m_stat_llvm_opt_time, m_stat_llvm_jit_time,
            m_llvm_local_mem / 1024);
    }
}



};  // namespace pvt
OSL_NAMESPACE_EXIT
