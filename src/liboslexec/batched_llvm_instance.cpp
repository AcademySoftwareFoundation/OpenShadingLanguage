// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <bitset>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/functional/hash.hpp>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/plugin.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/timer.h>

#include <OSL/batched_shaderglobals.h>
#include <OSL/batched_texture.h>

#include <llvm/IR/Constant.h>

#include "../liboslcomp/oslcomp_pvt.h"
#include "batched_backendllvm.h"
#include "oslexec_pvt.h"

// Create extrenal declarations for all built-in funcs we may call from LLVM
#define DECL(name, signature) extern "C" void name();
#include "builtindecl.h"
#undef DECL


/*
This whole file is concerned with taking our post-optimized OSO
intermediate code and translating it into LLVM IR code so we can JIT it
and run it directly, for an expected huge speed gain over running our
interpreter.

Schematically, we want to create code that resembles the following:

    // Assume 2 layers. 
    struct GroupData_1 {
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

    // Name of layer entry is $layer_ID
    void $layer_0 (ShaderGlobals *sg, GroupData_1 *group)
    {
        // Declare locals, temps, constants, params with known values.
        // Make them all look like stack memory locations:
        float *x = alloca (sizeof(float));
        // ...and so on for all the other locals & temps...

        // then run the shader body:
        *x = sg->u * group->param_0_bar;
        group->param_1_foo = *x;
    }

    void $layer_1 (ShaderGlobals *sg, GroupData_1 *group)
    {
        // Because we need the outputs of layer 0 now, we call it if it
        // hasn't already run:
        if (! group->layer_run[0]) {
            group->layer_run[0] = 1;
            $layer_0 (sg, group);    // because we need its outputs
        }
        *y = sg->u * group->$param_1_bar;
    }

    void $group_1 (ShaderGlobals *sg, GroupData_1 *group)
    {
        group->layer_run[...] = 0;
        // Run just the unconditional layers

        if (! group->layer_run[1]) {
            group->layer_run[1] = 1;
            $layer_1 (sg, group);
        }
    }

*/

extern int osl_llvm_compiled_ops_size;
extern unsigned char osl_llvm_compiled_ops_block[];

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


static TypeSpec
possibly_wide_type_from_code(const char* code, int* advance, bool& is_uniform)
{
    // Codes leading with a W stand for "wide" and have varying non-uniform values
    int i = 0;
    if (code[0] == 'W') {
        is_uniform = false;
        ++i;
    } else {
        is_uniform = true;
    }

    TypeSpec t = OSLCompilerImpl::type_from_code(code + i, advance);

    if (advance)
        *advance += i;
    return t;
}

struct HelperFuncRecord {
    const char* argtypes;
    void (*function)();
    int vector_width;
    TargetISA target_isa;

    OSL_FORCEINLINE HelperFuncRecord(const char* argtypes_ = NULL,
                                     void (*function_)()   = NULL,
                                     int vector_width_     = 0,
                                     TargetISA target_isa_ = TargetISA::UNKNOWN)
        : argtypes(argtypes_)
        , function(function_)
        , vector_width(vector_width_)
        , target_isa(target_isa_)
    {
    }

    OSL_FORCEINLINE HelperFuncRecord(const HelperFuncRecord& other)
        : argtypes(other.argtypes)
        , function(other.function)
        , vector_width(other.vector_width)
        , target_isa(other.target_isa)
    {
    }
};

// As we will be using compile time const char * to populate the HelperFuncMap
// We avoid std::string (which would create a copy), and instead just store
// the const char *, however we must supply our own hashing and equality
// functors to the map to avoid default behavior of using the pointer vs.
// the string it points to, in C++20 std::string_view could be used
struct CStrHash {
    OSL_FORCEINLINE size_t operator()(const char* str) const
    {
        OSL_DASSERT(str != nullptr);
        size_t seed = 0;
        for (;;) {
            char c = *str;
            if (c == 0)
                break;
            boost::hash_combine<char>(seed, c);
            ++str;
        }
        return seed;
    }
};
struct CStrEquality {
    OSL_FORCEINLINE bool operator()(const char* lhs, const char* rhs) const
    {
        bool is_equal = (strcmp(lhs, rhs) == 0);
        return is_equal;
    }
};
typedef std::unordered_map<const char*, HelperFuncRecord, CStrHash, CStrEquality>
    HelperFuncMap;
static HelperFuncMap llvm_helper_function_map;
static atomic_int llvm_helper_function_map_initialized(0);
static spin_mutex llvm_helper_function_map_mutex;

static void
initialize_llvm_helper_function_map()
{
    if (llvm_helper_function_map_initialized)
        return;  // already done
    spin_lock lock(llvm_helper_function_map_mutex);
    if (llvm_helper_function_map_initialized)
        return;
#define DECL(name, signature) \
    llvm_helper_function_map[#name] = HelperFuncRecord(signature, name);
#include "builtindecl.h"
#undef DECL

    llvm_helper_function_map_initialized = 1;
}

struct NameAndSignature {
    const char* name;
    const char* signature;
};

// As we don't now the sizeof ConcreteT::library_functions until specialization,
// we need a helper template function to defer its resolution
template<typename ConcreteT>
static void
init_wide_function_map(const ConcreteT&, ShadingSystemImpl& shadingsys)
{
    static atomic_int is_initialized(0);

    if (is_initialized)
        return;  // already done
    spin_lock lock(llvm_helper_function_map_mutex);
    if (is_initialized)
        return;

    const char* shared_lib_ext  = OIIO::Plugin::plugin_extension();
    std::string shared_lib_name = std::string("lib_")
                                  + ConcreteT::library_selector_string
                                  + "oslexec." + shared_lib_ext;
    //std::cout << ">>>Attempting to open shared lib:  " << shared_lib_name.c_str() << std::endl;

    std::string filename = OIIO::Filesystem::searchpath_find(
        shared_lib_name, shadingsys.library_searchpath_dirs());

    // TODO: consider trying to open it even if searchpath_find failed, so that LD_LIBRARY_PATH has a chance
    if (filename.empty()) {
        shadingsys.errorf(
            "%s could not be found along the attribute \"searchpath:library\" of \"%s\"",
            shared_lib_name.c_str(), shadingsys.library_searchpath().c_str());
        // Something later will ASSERT/Fail now, we can't really continue successfully
        return;
    }


    auto shared_lib = OIIO::Plugin::open(filename, /*global=*/false);
    if (shared_lib == 0) {
        shadingsys.errorf("%s could not be loaded with error \"%s\"",
                          filename.c_str(), OIIO::Plugin::geterror().c_str());
        // Something later will ASSERT/Fail, now we can't really continue successfully
        return;
    }

    typedef void (*FunctionPtr)();

    for (const auto& name_and_sig : ConcreteT::library_functions) {
        //std::cout << ">>>Attempting to getsym " << name_and_sig.name << std::endl;
        FunctionPtr function_pointer = reinterpret_cast<FunctionPtr>(
            OIIO::Plugin::getsym(shared_lib, name_and_sig.name,
                                 /*report_error*/ true));
        if (function_pointer == nullptr) {
            std::cout << ">>>Failed attempting to getsym " << name_and_sig.name
                      << std::endl
                      << "OIIO::Plugin::geterror()="
                      << OIIO::Plugin::geterror().c_str();
            ASSERT(
                0
                && "Unable to find precompiled OSL library function in shared library.  This indicates a build/configuration problem.  We can't continue");
        }
        llvm_helper_function_map[name_and_sig.name]
            = HelperFuncRecord(name_and_sig.signature, function_pointer,
                               ConcreteT::width, ConcreteT::isa);
    }

    is_initialized = 1;
}

template<int WidthT, TargetISA IsaT>
class ConcreteTargetLibraryHelper final
    : public BatchedBackendLLVM::TargetLibraryHelper {
public:
    ConcreteTargetLibraryHelper() {}
    ~ConcreteTargetLibraryHelper() final {}

    static constexpr int width     = WidthT;
    static constexpr TargetISA isa = IsaT;

    // Specialize instances for each supported Width and IsaT combo
    static const NameAndSignature library_functions[];
    static const char* library_selector_string;

    void init_function_map(ShadingSystemImpl& shadingsys) const final
    {
        init_wide_function_map(*this, shadingsys);
    }

    const char* library_selector() const final
    {
        return library_selector_string;
    }
};

// Specialize ConcreteTargetLibraryHelper<>::library_functions and
// ConcreteTargetLibraryHelper<>::library_selector_string for each
// WidthT and TargetISA that we are building a shared library for.
// To identify these shared libraries, the build system should define:
// __OSL_SUPPORTS_B##WidthT##_##TargetISA
// You will see conditional compilation around specilization below...

// NOTE:  Because builtindecl_wide_xmacro.h passes macros through the name
// parameter of DECL(name,signature), we must have a layer of indirection
// to allow those macros to expand.  Thus the use of DECL_INDIRECT.
#ifdef __OSL_SUPPORTS_B16_AVX512
template<>
const NameAndSignature
    ConcreteTargetLibraryHelper<16, TargetISA::AVX512>::library_functions[]
    = {
#    define DECL_INDIRECT(name, signature) \
        NameAndSignature { #name, signature },
#    define DECL(name, signature) DECL_INDIRECT(name, signature)
#    define __OSL_WIDTH           16
#    define __OSL_TARGET_ISA      AVX512
#    include "builtindecl_wide_xmacro.h"
#    include "wide/define_opname_macros.h"
#    include "wide/undef_opname_macros.h"
#    undef __OSL_TARGET_ISA
#    undef __OSL_WIDTH
#    undef DECL
#    undef DECL_INDIRECT
      };
template<>
const char*
    ConcreteTargetLibraryHelper<16, TargetISA::AVX512>::library_selector_string
    = "b16_AVX512_";
#endif

#ifdef __OSL_SUPPORTS_B16_AVX512_NOFMA
template<>
const NameAndSignature
    ConcreteTargetLibraryHelper<16, TargetISA::AVX512_noFMA>::library_functions[]
    = {
#    define DECL_INDIRECT(name, signature) \
        NameAndSignature { #name, signature },
#    define DECL(name, signature) DECL_INDIRECT(name, signature)
#    define __OSL_WIDTH           16
#    define __OSL_TARGET_ISA      AVX512_noFMA
#    include "builtindecl_wide_xmacro.h"
#    include "wide/define_opname_macros.h"
#    include "wide/undef_opname_macros.h"
#    undef __OSL_TARGET_ISA
#    undef __OSL_WIDTH
#    undef DECL
#    undef DECL_INDIRECT
      };
template<>
const char* ConcreteTargetLibraryHelper<
    16, TargetISA::AVX512_noFMA>::library_selector_string
    = "b16_AVX512_noFMA_";
#endif

#ifdef __OSL_SUPPORTS_B8_AVX512
template<>
const NameAndSignature
    ConcreteTargetLibraryHelper<8, TargetISA::AVX512>::library_functions[]
    = {
#    define DECL_INDIRECT(name, signature) \
        NameAndSignature { #name, signature },
#    define DECL(name, signature) DECL_INDIRECT(name, signature)
#    define __OSL_WIDTH           8
#    define __OSL_TARGET_ISA      AVX512
#    include "builtindecl_wide_xmacro.h"
#    include "wide/define_opname_macros.h"
#    include "wide/undef_opname_macros.h"
#    undef __OSL_TARGET_ISA
#    undef __OSL_WIDTH
#    undef DECL
#    undef DECL_INDIRECT
      };
template<>
const char*
    ConcreteTargetLibraryHelper<8, TargetISA::AVX512>::library_selector_string
    = "b8_AVX512_";
#endif

#ifdef __OSL_SUPPORTS_B8_AVX512_NOFMA
template<>
const NameAndSignature
    ConcreteTargetLibraryHelper<8, TargetISA::AVX512_noFMA>::library_functions[]
    = {
#    define DECL_INDIRECT(name, signature) \
        NameAndSignature { #name, signature },
#    define DECL(name, signature) DECL_INDIRECT(name, signature)
#    define __OSL_WIDTH           8
#    define __OSL_TARGET_ISA      AVX512_noFMA
#    include "builtindecl_wide_xmacro.h"
#    include "wide/define_opname_macros.h"
#    include "wide/undef_opname_macros.h"
#    undef __OSL_TARGET_ISA
#    undef __OSL_WIDTH
#    undef DECL
#    undef DECL_INDIRECT
      };
template<>
const char* ConcreteTargetLibraryHelper<
    8, TargetISA::AVX512_noFMA>::library_selector_string
    = "b8_AVX512_noFMA_";
#endif

#ifdef __OSL_SUPPORTS_B8_AVX2
template<>
const NameAndSignature
    ConcreteTargetLibraryHelper<8, TargetISA::AVX2>::library_functions[]
    = {
#    define DECL_INDIRECT(name, signature) \
        NameAndSignature { #name, signature },
#    define DECL(name, signature) DECL_INDIRECT(name, signature)
#    define __OSL_WIDTH           8
#    define __OSL_TARGET_ISA      AVX2
#    include "builtindecl_wide_xmacro.h"
#    include "wide/define_opname_macros.h"
#    include "wide/undef_opname_macros.h"
#    undef __OSL_TARGET_ISA
#    undef __OSL_WIDTH
#    undef DECL
#    undef DECL_INDIRECT
      };
template<>
const char*
    ConcreteTargetLibraryHelper<8, TargetISA::AVX2>::library_selector_string
    = "b8_AVX2_";
#endif

#ifdef __OSL_SUPPORTS_B8_AVX2_NOFMA
template<>
const NameAndSignature
    ConcreteTargetLibraryHelper<8, TargetISA::AVX2_noFMA>::library_functions[]
    = {
#    define DECL_INDIRECT(name, signature) \
        NameAndSignature { #name, signature },
#    define DECL(name, signature) DECL_INDIRECT(name, signature)
#    define __OSL_WIDTH           8
#    define __OSL_TARGET_ISA      AVX2_noFMA
#    include "builtindecl_wide_xmacro.h"
#    include "wide/define_opname_macros.h"
#    include "wide/undef_opname_macros.h"
#    undef __OSL_TARGET_ISA
#    undef __OSL_WIDTH
#    undef DECL
#    undef DECL_INDIRECT
      };
template<>
const char*
    ConcreteTargetLibraryHelper<8, TargetISA::AVX2_noFMA>::library_selector_string
    = "b8_AVX2_noFMA_";
#endif

#ifdef __OSL_SUPPORTS_B8_AVX
template<>
const NameAndSignature
    ConcreteTargetLibraryHelper<8, TargetISA::AVX>::library_functions[]
    = {
#    define DECL_INDIRECT(name, signature) \
        NameAndSignature { #name, signature },
#    define DECL(name, signature) DECL_INDIRECT(name, signature)
#    define __OSL_WIDTH           8
#    define __OSL_TARGET_ISA      AVX
#    include "builtindecl_wide_xmacro.h"
#    include "wide/define_opname_macros.h"
#    include "wide/undef_opname_macros.h"
#    undef __OSL_TARGET_ISA
#    undef __OSL_WIDTH
#    undef DECL
#    undef DECL_INDIRECT
      };
template<>
const char*
    ConcreteTargetLibraryHelper<8, TargetISA::AVX>::library_selector_string
    = "b8_AVX_";
#endif

std::unique_ptr<BatchedBackendLLVM::TargetLibraryHelper>
BatchedBackendLLVM::TargetLibraryHelper::build(int vector_width,
                                               TargetISA target_isa)
{
    typedef std::unique_ptr<BatchedBackendLLVM::TargetLibraryHelper> RetType;
    switch (vector_width) {
    case 16:
        switch (target_isa) {
#ifdef __OSL_SUPPORTS_B16_AVX512
        case TargetISA::AVX512:
            return RetType(
                new ConcreteTargetLibraryHelper<16, TargetISA::AVX512>());
#endif
#ifdef __OSL_SUPPORTS_B16_AVX512_NOFMA
        case TargetISA::AVX512_noFMA:
            return RetType(
                new ConcreteTargetLibraryHelper<16, TargetISA::AVX512_noFMA>());
#endif
        default:
            OSL_ASSERT(0 && "unsupported target ISA for vector width of 16");
        }
    case 8:
        switch (target_isa) {
#ifdef __OSL_SUPPORTS_B8_AVX512
        case TargetISA::AVX512:
            return RetType(
                new ConcreteTargetLibraryHelper<8, TargetISA::AVX512>());
#endif
#ifdef __OSL_SUPPORTS_B8_AVX512_NOFMA
        case TargetISA::AVX512_noFMA:
            return RetType(
                new ConcreteTargetLibraryHelper<8, TargetISA::AVX512_noFMA>());
#endif
#ifdef __OSL_SUPPORTS_B8_AVX2
        case TargetISA::AVX2:
            return RetType(
                new ConcreteTargetLibraryHelper<8, TargetISA::AVX2>());
#endif
#ifdef __OSL_SUPPORTS_B8_AVX2_NOFMA
        case TargetISA::AVX2_noFMA:
            return RetType(
                new ConcreteTargetLibraryHelper<8, TargetISA::AVX2_noFMA>());
#endif
#ifdef __OSL_SUPPORTS_B8_AVX
        case TargetISA::AVX:
            return RetType(
                new ConcreteTargetLibraryHelper<8, TargetISA::AVX>());
#endif
        default:
            OSL_ASSERT(0 && "unsupported target ISA for vector width of 8");
        }
    default: OSL_ASSERT(0 && "unsupported vector width");
    }
    return nullptr;
}


static void*
helper_function_lookup(const std::string& name)
{
    OSL_DEV_ONLY(std::cout << "helper_function_lookup (" << name << ")"
                           << std::endl);
    HelperFuncMap::const_iterator i = llvm_helper_function_map.find(
        name.c_str());
    if (i == llvm_helper_function_map.end()) {
        // built-in functions like memset wouldn't be in this lookup
        //std::cout << "DIDN'T FIND helper_function_lookup (" << name << ")" << std::endl;
        //for(auto v:llvm_helper_function_map) {
        //    std::cout << "llvm_helper_function_map [" << v.first << "]" << std::endl;
        //}
        return NULL;
    }
    return (void*)i->second.function;
}



llvm::Type*
BatchedBackendLLVM::llvm_type_sg()
{
    // Create a type that defines the ShaderGlobals for LLVM IR.  This
    // absolutely MUST exactly match the ShaderGlobals struct in oslexec.h.
    if (m_llvm_type_sg)
        return m_llvm_type_sg;

    // Derivs look like arrays of 3 values
    llvm::Type* wide_float_deriv = llvm_wide_type(
        TypeDesc(TypeDesc::FLOAT, TypeDesc::SCALAR, 3));
    llvm::Type* wide_triple_deriv = llvm_wide_type(
        TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, 3));

    llvm::Type* vp      = (llvm::Type*)ll.type_void_ptr();
    llvm::Type* wide_vp = (llvm::Type*)ll.type_wide_void_ptr();

    std::vector<llvm::Type*> sg_types;

    // Uniform values of the batch
    sg_types.push_back(vp);             // opaque renderstate*
    sg_types.push_back(vp);             // opaque tracedata*
    sg_types.push_back(vp);             // opaque objdata*
    sg_types.push_back(vp);             // ShadingContext*
    sg_types.push_back(vp);             // RendererServices*
    sg_types.push_back(vp);             // Ci
    sg_types.push_back(ll.type_int());  // raytype
    sg_types.push_back(ll.type_int());  // pad0
    sg_types.push_back(ll.type_int());  // pad1
    sg_types.push_back(ll.type_int());  // pad2


    // VaryingShaderGlobals of the batch
    sg_types.push_back(wide_triple_deriv);      // P, dPdx, dPdy
    sg_types.push_back(ll.type_wide_triple());  // dPdz
    sg_types.push_back(wide_triple_deriv);      // I, dIdx, dIdy
    sg_types.push_back(ll.type_wide_triple());  // N
    sg_types.push_back(ll.type_wide_triple());  // Ng
    sg_types.push_back(wide_float_deriv);       // u, dudx, dudy
    sg_types.push_back(wide_float_deriv);       // v, dvdx, dvdy
    sg_types.push_back(ll.type_wide_triple());  // dPdu
    sg_types.push_back(ll.type_wide_triple());  // dPdv
    sg_types.push_back(ll.type_wide_float());   // time
    sg_types.push_back(ll.type_wide_float());   // dtime
    sg_types.push_back(ll.type_wide_triple());  // dPdtime
    sg_types.push_back(wide_triple_deriv);      // Ps, dPsdx, dPsdy;

    sg_types.push_back(wide_vp);  // object2common
    sg_types.push_back(wide_vp);  // shader2common

    sg_types.push_back(ll.type_wide_float());  // surfacearea
    sg_types.push_back(ll.type_wide_int());    // flipHandedness
    sg_types.push_back(ll.type_wide_int());    // backfacing

    return m_llvm_type_sg = ll.type_struct(sg_types, "BatchedShaderGlobals",
                                           true /*is_packed*/);
}


llvm::Type*
BatchedBackendLLVM::llvm_type_batched_texture_options()
{
    // Create a type that defines the BatchedTextureOptions for LLVM IR.  This
    // absolutely MUST exactly match the BatchedTextureOptions struct in batched_texture.h.
    if (m_llvm_type_batched_texture_options)
        return m_llvm_type_batched_texture_options;

    llvm::Type* vp = (llvm::Type*)ll.type_void_ptr();

    std::vector<llvm::Type*> sg_types;

    // Varying values of the batch
    sg_types.push_back(ll.type_wide_float());  // sblur
    sg_types.push_back(ll.type_wide_float());  // tblur
    sg_types.push_back(ll.type_wide_float());  // rblur
    sg_types.push_back(ll.type_wide_float());  // swidth
    sg_types.push_back(ll.type_wide_float());  // twidth
    sg_types.push_back(ll.type_wide_float());  // rwidth

    // Uniform values of the batch
    sg_types.push_back(ll.type_int());                 // firstchannel
    sg_types.push_back(ll.type_int());                 // subimage
    sg_types.push_back(vp);                            // subimagename
    sg_types.push_back(ll.type_int());                 // swrap
    sg_types.push_back(ll.type_int());                 // twrap
    sg_types.push_back(ll.type_int());                 // rwrap
    sg_types.push_back(ll.type_int());                 // mipmode
    sg_types.push_back(ll.type_int());                 // interpmode
    sg_types.push_back(ll.type_int());                 // anisotropic
    sg_types.push_back(ll.type_int());                 // conservative_filter
    sg_types.push_back(ll.type_float());               // fill
    sg_types.push_back(ll.type_ptr(ll.type_float()));  // missingcolor

    // Private internal data
    sg_types.push_back(ll.type_int());  // envlayout

    m_llvm_type_batched_texture_options
        = ll.type_struct(sg_types, "BatchedTextureOptions",
                         false /*is_packed*/);


#if 0 && defined(OSL_DEV)
    std::cout << std::endl << std::endl << "llvm's data layout of BatchedTextureOptions" << std::endl;
    ll.dump_struct_data_layout(llvm_type_batched_texture_options());
#endif

    {
        std::vector<unsigned int> offset_by_index;
        switch (m_width) {
        case 8:
            build_offsets_of_BatchedTextureOptions<8>(offset_by_index);
            break;
        case 16:
            build_offsets_of_BatchedTextureOptions<16>(offset_by_index);
            break;
        default:
            OSL_ASSERT(
                0
                && "Unsupported width of batch.  Only widths 4, 8, and 16 are allowed");
            break;
        };

        ll.validate_struct_data_layout(m_llvm_type_batched_texture_options,
                                       offset_by_index);
        // std::cout<<"After texture validation"<<std::endl;
    }

    return m_llvm_type_batched_texture_options;
}


llvm::Type*
BatchedBackendLLVM::llvm_type_batched_trace_options()
{
    // Create a type that defines the BatchedTraceOptions for LLVM IR.  This
    // absolutely MUST exactly match the BatchedTraceOptions struct in batched_texture.h.
    if (m_llvm_type_batched_trace_options)
        return m_llvm_type_batched_trace_options;

    std::vector<llvm::Type*> sg_types;

    // Uniform values of the batch
    sg_types.push_back(ll.type_float());  // mindist
    sg_types.push_back(ll.type_float());  // maxdist
    sg_types.push_back(ll.type_int());    // shade
    sg_types.push_back(
        reinterpret_cast<llvm::Type*>(ll.type_string()));  // traceset

    m_llvm_type_batched_trace_options = ll.type_struct(sg_types, "TraceOptions",
                                                       false /*is_packed*/);

    {
        std::vector<unsigned int> offset_by_index;

        offset_by_index.push_back(
            offsetof(RendererServices::TraceOpt, mindist));
        offset_by_index.push_back(
            offsetof(RendererServices::TraceOpt, maxdist));
        offset_by_index.push_back(offsetof(RendererServices::TraceOpt, shade));
        offset_by_index.push_back(
            offsetof(RendererServices::TraceOpt, traceset));
        //        std::cout<<"Offset vec size is "<<offset_by_index.size()<<std::endl;
        //        std::cout<<"Offset by index size is "<<offset_by_index.size()<<std::endl;
        //        std::cout<<"Offset_by_index[0] "<<offset_by_index[0]<<std::endl;
        //        std::cout<<"Offset_by_index[1] "<<offset_by_index[1]<<std::endl;
        //        std::cout<<"Offset_by_index[2] "<<offset_by_index[2]<<std::endl;
        //        std::cout<<"Offset_by_index[3] "<<offset_by_index[3]<<std::endl;

        ll.validate_struct_data_layout(m_llvm_type_batched_trace_options,
                                       offset_by_index);
    }

    return m_llvm_type_batched_trace_options;
}


llvm::Type*
BatchedBackendLLVM::llvm_type_sg_ptr()
{
    return ll.type_ptr(llvm_type_sg());
}



llvm::Type*
BatchedBackendLLVM::llvm_type_groupdata()
{
    // If already computed, return it
    if (m_llvm_type_groupdata)
        return m_llvm_type_groupdata;

    std::vector<llvm::Type*> fields;
    int offset = 0;
    int order  = 0;

    if (llvm_debug() >= 2)
        std::cout << "Group param struct:\n";

    // First, add the array that tells if each layer has run.  But only make
    // slots for the layers that may be called/used.
    if (llvm_debug() >= 2)
        std::cout << "  layers run flags: " << m_num_used_layers
                  << " at offset " << offset << "\n";
    // The next item in the data structure has 64 byte alignment, so we need to move our offset to a 64 byte alignment
    // Round up to a 64 bit boundary
    int sz = 16 * ((m_num_used_layers + 15) / 16);
    OSL_ASSERT(sz * sizeof(int) % 16 == 0);
    fields.push_back(ll.type_array(ll.type_int(), sz));
    offset += sz * sizeof(int);
    ++order;

    // Now add the array that tells which userdata have been initialized,
    // and the space for the userdata values.
    int nuserdata = (int)group().m_userdata_names.size();
    if (nuserdata) {
        if (llvm_debug() >= 2)
            std::cout << "  userdata initialized flags: " << nuserdata
                      << " at offset " << offset << ", field " << order << "\n";
        ustring* names = &group().m_userdata_names[0];
        OSL_DEV_ONLY(std::cout << "USERDATA " << *names << std::endl);
        TypeDesc* types = &group().m_userdata_types[0];
        int* offsets    = &group().m_userdata_offsets[0];
        int sz          = nuserdata;
        fields.push_back(ll.type_array(ll.type_int(), sz));
        offset += nuserdata * sizeof(int);
        ++order;
        for (int i = 0; i < nuserdata; ++i) {
            TypeDesc type = types[i];
            // TODO: why do we always make deriv room? Do we not know
            int n         = type.numelements() * 3;  // always make deriv room
            type.arraylen = n;
            fields.push_back(llvm_wide_type(type));
            // Alignment
            int align = type.basesize() * m_width;
            offset    = OIIO::round_to_multiple_of_pow2(offset, align);
            if (llvm_debug() >= 2) {
                std::cout << "  userdata ";
                if (names[i] != nullptr) {
                    std::cout << names[i];
                } else {
                    std::cout << i;
                }
                std::cout << ' ' << type << ", field " << order << ", offset "
                          << offset << std::endl;
            }
            offsets[i] = offset;
            offset += int(type.size()) * m_width;
            ++order;
        }
    }

    // For each layer in the group, add entries for all params that are
    // connected or interpolated, and output params.  Also mark those
    // symbols with their offset within the group struct.
    m_param_order_map.clear();
    for (int layer = 0; layer < group().nlayers(); ++layer) {
        ShaderInstance* inst = group()[layer];
        // TODO:  Does anything bad happen from not skipping unused layers?
        // We wanted space for default parameters to still be
        // part of group data so we have a place to create a wide version
        // So we choose to always have a run function for a layer
        // just to broadcast out the scalar default value.
        // TODO: Optimize to only run unused layers once, shouldn't
        // need to be run again as nothing should overwrite the values.
        FOREACH_PARAM(Symbol & sym, inst)
        {
            TypeSpec ts = sym.typespec();
            if (ts.is_structure())  // skip the struct symbol itself
                continue;
            const int arraylen  = std::max(1, sym.typespec().arraylength());
            const int derivSize = (sym.has_derivs() ? 3 : 1);
            ts.make_array(arraylen * derivSize);
            fields.push_back(llvm_wide_type(ts));

            // Alignment
            // TODO:  this isn't quite right, cant rely on batch size to == ISA SIMD requirements
            size_t align = sym.typespec().is_closure_based()
                               ? sizeof(void*)
                               : sym.typespec().simpletype().basesize()
                                     * m_width;
            if (offset & (align - 1))
                offset += align - (offset & (align - 1));
            if (llvm_debug() >= 2)
                std::cout << "  " << inst->layername() << " (" << inst->id()
                          << ") " << sym.mangled() << " " << ts.c_str()
                          << ", field " << order << ", size "
                          << derivSize * int(sym.size()) << ", offset "
                          << offset << std::endl;
            sym.wide_dataoffset((int)offset);
            offset += derivSize * int(sym.size()) * m_width;

            m_param_order_map[&sym] = order;
            ++order;
        }
    }
    group().llvm_groupdata_wide_size(offset);
    if (llvm_debug() >= 2)
        std::cout << " Group struct had " << order << " fields, total size "
                  << offset << "\n\n";

    std::string groupdataname
        = Strutil::sprintf("Groupdata_%llu",
                           (long long unsigned int)group().name().hash());
    m_llvm_type_groupdata = ll.type_struct(fields, groupdataname,
                                           false /*is_packed*/);

    return m_llvm_type_groupdata;
}



llvm::Type*
BatchedBackendLLVM::llvm_type_groupdata_ptr()
{
    return ll.type_ptr(llvm_type_groupdata());
}



llvm::Type*
BatchedBackendLLVM::llvm_type_closure_component()
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
BatchedBackendLLVM::llvm_type_closure_component_ptr()
{
    return ll.type_ptr(llvm_type_closure_component());
}



void
BatchedBackendLLVM::llvm_assign_initial_value(
    const Symbol& sym, llvm::Value* llvm_initial_shader_mask_value, bool force)
{
    // Don't write over connections!  Connection values are written into
    // our layer when the earlier layer is run, as part of its code.  So
    // we just don't need to initialize it here at all.
    if (!force && sym.valuesource() == Symbol::ConnectedVal
        && !sym.typespec().is_closure_based())
        return;
    if (sym.typespec().is_closure_based() && sym.symtype() == SymTypeGlobal)
        return;

    int arraylen = std::max(1, sym.typespec().arraylength());

    // Closures need to get their storage before anything can be
    // assigned to them.  Unless they are params, in which case we took
    // care of it in the group entry point.
    if (sym.typespec().is_closure_based() && sym.symtype() != SymTypeParam
        && sym.symtype() != SymTypeOutputParam) {
        llvm_assign_zero(sym);
        return;
    }

    if ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp)
        && shadingsys().debug_uninit()) {
        // Handle the "debug uninitialized values" case
        bool isarray   = sym.typespec().is_array();
        int alen       = isarray ? sym.typespec().arraylength() : 1;
        llvm::Value* u = NULL;
        if (sym.typespec().is_closure_based()) {
            // skip closures
        } else if (sym.typespec().is_float_based()) {
            u = sym.is_uniform()
                    ? ll.constant(std::numeric_limits<float>::quiet_NaN())
                    : ll.wide_constant(std::numeric_limits<float>::quiet_NaN());
        } else if (sym.typespec().is_int_based()) {
            // Because we allow temporaries and local results of comparison operations
            // to use the native bool type of i1, we can just skip initializing these
            // as they should always be assigned a value.
            // We can just interrogate the underlying llvm symbol to see if
            // it is a bool
            llvm::Value* llvmValue = llvm_get_pointer(sym);
            if (ll.llvm_typeof(llvmValue) != ll.type_ptr(ll.type_bool())
                && ll.llvm_typeof(llvmValue)
                       != ll.type_ptr(ll.type_wide_bool())) {
                u = sym.is_uniform()
                        ? ll.constant(std::numeric_limits<int>::min())
                        : ll.wide_constant(std::numeric_limits<int>::min());
            }
        } else if (sym.typespec().is_string_based()) {
            u = sym.is_uniform()
                    ? ll.constant(Strings::uninitialized_string)
                    : ll.wide_constant(Strings::uninitialized_string);
        }
        if (u) {
            //std::cout << "Assigning uninit value to symbol=" << sym.name().c_str() << std::endl;
            for (int a = 0; a < alen; ++a) {
                llvm::Value* aval = isarray ? ll.constant(a) : NULL;
                for (int c = 0; c < (int)sym.typespec().aggregate(); ++c)
                    llvm_store_value(u, sym, 0, aval, c);
            }
        }
        return;
    }

    if ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp)
        && sym.typespec().is_string_based()) {
        // Strings are pointers.  Can't take any chance on leaving
        // local/tmp syms uninitialized.
        llvm_assign_zero(sym);
        return;  // we're done, the parts below are just for params
    }
    ASSERT_MSG(sym.symtype() == SymTypeParam
                   || sym.symtype() == SymTypeOutputParam,
               "symtype was %d, data type was %s", (int)sym.symtype(),
               sym.typespec().c_str());

    // Handle interpolated params by calling osl_bind_interpolated_param,
    // which will check if userdata is already retrieved, if not it will
    // call RendererServices::get_userdata to retrieve it. In either case,
    // it will return 1 if it put the userdata in the right spot (either
    // retrieved de novo or copied from a previous retrieval), or 0 if no
    // such userdata was available.
    llvm::BasicBlock* after_userdata_block = NULL;
    bool partial_userdata_mask_was_pushed  = false;
    LLVM_Util::ScopedMasking partial_data_masking_scope;
    if (!sym.lockgeom() && !sym.typespec().is_closure()
        && !(sym.symtype() == SymTypeOutputParam)) {
        ustring symname = sym.name();
        TypeDesc type   = sym.typespec().simpletype();

        int userdata_index = find_userdata_index(sym);
        OSL_ASSERT(userdata_index >= 0);

        // User connectable params must be varying
        OSL_ASSERT(sym.is_varying());
        std::vector<llvm::Value*> args;
        args.push_back(sg_void_ptr());
        args.push_back(ll.constant(symname));
#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
        args.push_back(ll.constant(inst()->layername()));
#endif
        args.push_back(ll.constant(type));
        args.push_back(
            ll.constant((int)group().m_userdata_derivs[userdata_index]));
        args.push_back(
            groupdata_field_ptr(2 + userdata_index));  // userdata data ptr
        args.push_back(ll.constant((int)sym.has_derivs()));
        args.push_back(llvm_void_ptr(sym));
        args.push_back(ll.constant(sym.derivsize() * m_width));
        args.push_back(ll.void_ptr(userdata_initialized_ref(userdata_index)));
        args.push_back(ll.constant(userdata_index));
        args.push_back(llvm_initial_shader_mask_value);
        llvm::Value* got_userdata
            = ll.call_function(build_name("bind_interpolated_param"), args);
        llvm::Value* got_userdata_mask = ll.int_as_mask(got_userdata);

        if (shadingsys().debug_nan() && type.basetype == TypeDesc::FLOAT) {
            // check for NaN/Inf for float-based types
            int ncomps          = type.numelements() * type.aggregate;
            llvm::Value* args[] = { ll.mask_as_int(ll.current_mask()),
                                    ll.constant(ncomps),
                                    llvm_void_ptr(sym),
                                    ll.constant((int)sym.has_derivs()),
                                    sg_void_ptr(),
                                    ll.constant(ustring(inst()->shadername())),
                                    ll.constant(0),
                                    ll.constant(sym.name()),
                                    ll.constant(0),
                                    ll.constant(ncomps),
                                    ll.constant("<get_userdata>") };
            ll.call_function(build_name(FuncSpec("naninf_check_offset")
                                            .arg_uniform(TypeDesc::TypeInt)
                                            .mask()),
                             args);
        }
        // We will enclose the subsequent initialization of default values
        // or init ops in an "if" so that the extra copies or code don't
        // happen if the userdata was retrieved.
        llvm::BasicBlock* partial_userdata_block = ll.new_basic_block(
            "partial_userdata");
        after_userdata_block  = ll.new_basic_block();
        llvm::Value* cond_val = ll.op_ne(got_userdata,
                                         ll.constant(true_mask_value()));
        ll.op_branch(cond_val, partial_userdata_block, after_userdata_block);

        // If we got no or partial user data, we need to mask out the lanes
        // that successfully got user data from the initops or default value
        // assignment
        ll.push_mask(
            got_userdata_mask, /* negate */
            true /*, absolute = false (not sure how it wouldn't be an absolute mask) */);
        partial_data_masking_scope = ll.create_masking_scope(/*enabled=*/true);
        partial_userdata_mask_was_pushed = true;
    }

    if (sym.has_init_ops() && sym.valuesource() == Symbol::DefaultVal) {
        // Forcing masking shouldn't be required here,
        // believe our discovery handled this correctly
        // as these are initialization op's that are being processed,
        // they should have corresponding require's masking entries,
        // unlike the rest of the copies/initialization going on here

        // Handle init ops.
        build_llvm_code(sym.initbegin(), sym.initend());
    } else {
        // We think the non-memcpy route is preferable as it give the compiler
        // a chance to optimize constant values Also memcpy would ignoring the
        // mask stack, which is problematic
        LLVM_Util::ScopedMasking render_output_masking_scope;
        if (sym.renderer_output()) {
            render_output_masking_scope = ll.create_masking_scope(
                /*enabled=*/true);
        }

        // Use default value
        int num_components = sym.typespec().simpletype().aggregate;
        TypeSpec elemtype  = sym.typespec().elementtype();
        for (int a = 0, c = 0; a < arraylen; ++a) {
            llvm::Value* arrind = sym.typespec().is_array() ? ll.constant(a)
                                                            : NULL;
            if (sym.typespec().is_closure_based())
                continue;
            for (int i = 0; i < num_components; ++i, ++c) {
                // Fill in the constant val
                llvm::Value* init_val = 0;
                if (elemtype.is_float_based())
                    init_val = ll.constant(((float*)sym.data())[c]);
                else if (elemtype.is_string())
                    init_val = ll.constant(((ustring*)sym.data())[c]);
                else if (elemtype.is_int())
                    init_val = ll.constant(((int*)sym.data())[c]);
                OSL_ASSERT(init_val);

                if (sym.is_uniform()) {
                    OSL_ASSERT(!sym.renderer_output()
                               && "All render outputs should be varying");
                    llvm_store_value(init_val, sym, 0, arrind, i);
                } else {
                    llvm::Value* wide_init_val = ll.wide_constant(
                        static_cast<llvm::Constant*>(init_val));
                    llvm_store_value(wide_init_val, sym, 0, arrind, i);
                }
            }
        }
        if (sym.has_derivs()) {
            llvm_zero_derivs(sym);
        }
    }

    if (partial_userdata_mask_was_pushed) {
        partial_data_masking_scope.release();
        ll.pop_mask();
    }

    if (after_userdata_block) {
        // If we enclosed the default initialization in an "if", jump to the
        // next basic block now.
        ll.op_branch(after_userdata_block);
    }
}



void
BatchedBackendLLVM::llvm_generate_debugnan(const Opcode& op)
{
    for (int i = 0; i < op.nargs(); ++i) {
        Symbol& sym(*opargsym(op, i));
        if (!op.argwrite(i))
            continue;
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT)
            continue;  // just check float-based types
        llvm::Value* ncomps = ll.constant(int(t.numelements() * t.aggregate));
        llvm::Value* offset = ll.constant(0);
        llvm::Value* ncheck = ncomps;
        BatchedBackendLLVM::TempScope temp_scope(*this);
        llvm::Value* loc_varying_offsets = nullptr;
        if (op.opname() == op_aassign) {
            // Special case -- array assignment -- only check one element
            OSL_ASSERT(i == 0 && "only arg 0 is written for aassign");
            Symbol& index_sym = *opargsym(op, 1);
            llvm::Value* ind  = llvm_load_value(index_sym);
            llvm::Value* agg  = index_sym.is_uniform()
                                   ? ll.constant(t.aggregate)
                                   : ll.wide_constant(t.aggregate);
            llvm::Value* scaled_offset = t.aggregate == 1 ? ind
                                                          : ll.op_mul(ind, agg);
            if (index_sym.is_uniform()) {
                offset = scaled_offset;
            } else {
                loc_varying_offsets
                    = getOrAllocateTemp(TypeSpec(TypeDesc::INT),
                                        false /*derivs*/, false /*is_uniform*/,
                                        false /*forceBool*/,
                                        std::string("nan check scaled indices:")
                                            + index_sym.name().c_str());
                ll.op_store(scaled_offset, loc_varying_offsets);
            }
            ncheck = ll.constant(t.aggregate);
        } else if (op.opname() == op_compassign) {
            // Special case -- component assignment -- only check one channel
            OSL_ASSERT(i == 0 && "only arg 0 is written for compassign");
            Symbol& index_sym = *opargsym(op, 1);
            if (index_sym.is_uniform()) {
                offset = llvm_load_value(index_sym);
            } else {
                loc_varying_offsets = llvm_get_pointer(index_sym);
            }
            ncheck = ll.constant(1);
        } else if (op.opname() == op_mxcompassign) {
            // Special case -- matrix component assignment -- only check one channel
            OSL_ASSERT(i == 0 && "only arg 0 is written for compassign");
            Symbol& row_sym             = *opargsym(op, 1);
            Symbol& col_sym             = *opargsym(op, 2);
            bool components_are_uniform = row_sym.is_uniform()
                                          && col_sym.is_uniform();

            llvm::Value* row_ind = llvm_load_value(row_sym, 0, 0,
                                                   TypeDesc::UNKNOWN,
                                                   components_are_uniform);
            llvm::Value* col_ind = llvm_load_value(col_sym, 0, 0,
                                                   TypeDesc::UNKNOWN,
                                                   components_are_uniform);

            llvm::Value* comp = ll.op_mul(row_ind, components_are_uniform
                                                       ? ll.constant(4)
                                                       : ll.wide_constant(4));
            comp              = ll.op_add(comp, col_ind);

            if (components_are_uniform) {
                offset = comp;
            } else {
                loc_varying_offsets
                    = getOrAllocateTemp(TypeSpec(TypeDesc::INT),
                                        false /*derivs*/, false /*is_uniform*/,
                                        false /*forceBool*/,
                                        std::string("nan check comp from row(")
                                            + row_sym.name().c_str() + ") col("
                                            + col_sym.name().c_str() + ")");
                ll.op_store(comp, loc_varying_offsets);
            }
            ncheck = ll.constant(1);
        }

        if (loc_varying_offsets != nullptr) {
            OSL_ASSERT(sym.is_varying());
            llvm::Value* args[] = { ll.mask_as_int(ll.current_mask()),
                                    ncomps,
                                    llvm_void_ptr(sym),
                                    ll.constant((int)sym.has_derivs()),
                                    sg_void_ptr(),
                                    ll.constant(op.sourcefile()),
                                    ll.constant(op.sourceline()),
                                    ll.constant(sym.name()),
                                    ll.void_ptr(loc_varying_offsets),
                                    ncheck,
                                    ll.constant(op.opname()) };
            ll.call_function(build_name(FuncSpec("naninf_check_offset")
                                            .arg_varying(TypeDesc::TypeInt)
                                            .mask()),
                             args);

        } else {
            if (sym.is_uniform()) {
                llvm::Value* args[] = { ncomps,
                                        llvm_void_ptr(sym),
                                        ll.constant((int)sym.has_derivs()),
                                        sg_void_ptr(),
                                        ll.constant(op.sourcefile()),
                                        ll.constant(op.sourceline()),
                                        ll.constant(sym.name()),
                                        offset,
                                        ncheck,
                                        ll.constant(op.opname()) };
                ll.call_function(build_name("naninf_check"), args);
            } else {
                llvm::Value* args[] = { ll.mask_as_int(ll.current_mask()),
                                        ncomps,
                                        llvm_void_ptr(sym),
                                        ll.constant((int)sym.has_derivs()),
                                        sg_void_ptr(),
                                        ll.constant(op.sourcefile()),
                                        ll.constant(op.sourceline()),
                                        ll.constant(sym.name()),
                                        offset,
                                        ncheck,
                                        ll.constant(op.opname()) };
                ll.call_function(build_name(FuncSpec("naninf_check_offset")
                                                .arg_uniform(TypeDesc::TypeInt)
                                                .mask()),
                                 args);
            }
        }
    }
}



void
BatchedBackendLLVM::llvm_generate_debug_uninit(const Opcode& op)
{
    if (op.opname() == op_useparam) {
        // Don't check the args of a useparam before the op; they are by
        // definition potentially net yet set before the useparam action
        // itself puts values into them. Checking them for uninitialized
        // values will result in false positives.
        return;
    }
    for (int i = 0; i < op.nargs(); ++i) {
        Symbol& sym(*opargsym(op, i));
        if (!op.argread(i))
            continue;
        if (sym.typespec().is_closure_based())
            continue;
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT && t.basetype != TypeDesc::INT
            && t.basetype != TypeDesc::STRING)
            continue;  // just check float, int, string based types

        // Because we allow temporaries and local results of comparison operations
        // to use the native bool type of i1, we can just skip checking these
        // as they should always be assigned a value.
        // We can just interrogate the underlying llvm symbol to see if
        // it is a bool
        llvm::Value* llvmValue = llvm_get_pointer(sym);
        if (ll.llvm_typeof(llvmValue) == ll.type_ptr(ll.type_bool())
            || ll.llvm_typeof(llvmValue) == ll.type_ptr(ll.type_wide_bool())) {
            continue;
        }

        llvm::Value* ncheck = ll.constant(int(t.numelements() * t.aggregate));
        llvm::Value* offset = ll.constant(0);
        BatchedBackendLLVM::TempScope temp_scope(*this);
        llvm::Value* loc_varying_offsets = nullptr;

        // Some special cases...
        if (op.opname() == Strings::op_for && i == 0) {
            // The first argument of 'for' is the condition temp, but
            // note that it may not have had its initializer run yet, so
            // don't generate uninit test code for it.
            continue;
        }
        if (op.opname() == Strings::op_dowhile && i == 0) {
            // The first argument of 'dowhile' is the condition temp, but
            // it most likely its initializer run yet.
            // Unless there is no "condition" code block, in that
            // case we should still
            if (op.jump(0) != op.jump(1))
                continue;
        }
        if (op.opname() == op_aref && i == 1) {
            // Special case -- array reference -- only check one element
            Symbol& index_sym = *opargsym(op, 2);
            llvm::Value* ind  = llvm_load_value(index_sym);

            llvm::Value* agg = index_sym.is_uniform()
                                   ? ll.constant(t.aggregate)
                                   : ll.wide_constant(t.aggregate);
            llvm::Value* scaled_offset = t.aggregate == 1 ? ind
                                                          : ll.op_mul(ind, agg);
            if (index_sym.is_uniform()) {
                offset = scaled_offset;
            } else {
                loc_varying_offsets = getOrAllocateTemp(
                    TypeSpec(TypeDesc::INT), false /*derivs*/,
                    false /*is_uniform*/, false /*forceBool*/,
                    std::string("uninit check scaled indices:")
                        + index_sym.name().c_str());
                ll.op_store(scaled_offset, loc_varying_offsets);
            }
            ncheck = ll.constant(t.aggregate);

        } else if (op.opname() == op_compref && i == 1) {
            // Special case -- component reference -- only check one channel
            Symbol& index_sym = *opargsym(op, 2);
            if (index_sym.is_uniform()) {
                offset = llvm_load_value(index_sym);
            } else {
                loc_varying_offsets = llvm_get_pointer(index_sym);
            }
            ncheck = ll.constant(1);
        } else if (op.opname() == op_mxcompref && i == 1) {
            // Special case -- matrix component reference -- only check one channel
            Symbol& row_sym             = *opargsym(op, 2);
            Symbol& col_sym             = *opargsym(op, 3);
            bool components_are_uniform = row_sym.is_uniform()
                                          && col_sym.is_uniform();

            llvm::Value* row_ind = llvm_load_value(row_sym, 0, 0,
                                                   TypeDesc::UNKNOWN,
                                                   components_are_uniform);
            llvm::Value* col_ind = llvm_load_value(col_sym, 0, 0,
                                                   TypeDesc::UNKNOWN,
                                                   components_are_uniform);

            llvm::Value* comp = ll.op_mul(row_ind, components_are_uniform
                                                       ? ll.constant(4)
                                                       : ll.wide_constant(4));
            comp              = ll.op_add(comp, col_ind);

            if (components_are_uniform) {
                offset = comp;
            } else {
                loc_varying_offsets = getOrAllocateTemp(
                    TypeSpec(TypeDesc::INT), false /*derivs*/,
                    false /*is_uniform*/, false /*forceBool*/,
                    std::string("uninit check comp from row(")
                        + row_sym.name().c_str() + ") col("
                        + col_sym.name().c_str() + ")");
                ll.op_store(comp, loc_varying_offsets);
            }
            ncheck = ll.constant(1);
        }

        if (loc_varying_offsets != nullptr) {
            llvm::Value* args[] = { ll.mask_as_int(ll.current_mask()),
                                    ll.constant(t),
                                    llvm_void_ptr(sym),
                                    sg_void_ptr(),
                                    ll.constant(op.sourcefile()),
                                    ll.constant(op.sourceline()),
                                    ll.constant(group().name()),
                                    ll.constant(layer()),
                                    ll.constant(inst()->layername()),
                                    ll.constant(inst()->shadername().c_str()),
                                    ll.constant(int(&op - &inst()->ops()[0])),
                                    ll.constant(op.opname()),
                                    ll.constant(i),
                                    ll.constant(sym.name()),
                                    ll.void_ptr(loc_varying_offsets),
                                    ncheck };

            if (sym.is_uniform()) {
                ll.call_function(build_name(
                                     FuncSpec("uninit_check_values_offset")
                                         .arg_uniform(TypeDesc::PTR)
                                         .arg_varying(TypeDesc::TypeInt)
                                         .mask()),
                                 args);
            } else {
                ll.call_function(build_name(
                                     FuncSpec("uninit_check_values_offset")
                                         .arg_varying(TypeDesc::PTR)
                                         .arg_varying(TypeDesc::TypeInt)
                                         .mask()),
                                 args);
            }
        } else {
            if (sym.is_uniform()) {
                llvm::Value* args[]
                    = { ll.constant(t),
                        llvm_void_ptr(sym),
                        sg_void_ptr(),
                        ll.constant(op.sourcefile()),
                        ll.constant(op.sourceline()),
                        ll.constant(group().name()),
                        ll.constant(layer()),
                        ll.constant(inst()->layername()),
                        ll.constant(inst()->shadername().c_str()),
                        ll.constant(int(&op - &inst()->ops()[0])),
                        ll.constant(op.opname()),
                        ll.constant(i),
                        ll.constant(sym.name()),
                        offset,
                        ncheck };
                ll.call_function(build_name(
                                     FuncSpec("uninit_check_values_offset")
                                         .arg_uniform(TypeDesc::PTR)
                                         .arg_uniform(TypeDesc::TypeInt)),
                                 args);
            } else {
                llvm::Value* args[]
                    = { ll.mask_as_int(ll.current_mask()),
                        ll.constant(t),
                        llvm_void_ptr(sym),
                        sg_void_ptr(),
                        ll.constant(op.sourcefile()),
                        ll.constant(op.sourceline()),
                        ll.constant(group().name()),
                        ll.constant(layer()),
                        ll.constant(inst()->layername()),
                        ll.constant(inst()->shadername().c_str()),
                        ll.constant(int(&op - &inst()->ops()[0])),
                        ll.constant(op.opname()),
                        ll.constant(i),
                        ll.constant(sym.name()),
                        offset,
                        ncheck };
                ll.call_function(build_name(
                                     FuncSpec("uninit_check_values_offset")
                                         .arg_varying(TypeDesc::PTR)
                                         .arg_uniform(TypeDesc::TypeInt)
                                         .mask()),
                                 args);
            }
        }
    }
}


void
BatchedBackendLLVM::llvm_generate_debug_op_printf(const Opcode& op)
{
    std::ostringstream msg;
    msg << op.sourcefile() << ':' << op.sourceline() << ' ' << op.opname();
    for (int i = 0; i < op.nargs(); ++i)
        msg << ' ' << opargsym(op, i)->mangled();
    llvm_gen_debug_printf(msg.str());
}


bool
BatchedBackendLLVM::build_llvm_code(int beginop, int endop,
                                    llvm::BasicBlock* bb)
{
    OSL_DEV_ONLY(std::cout << "build_llvm_code : beginop=" << beginop
                           << " endop=" << endop << " bb=" << bb << std::endl);
    if (bb)
        ll.set_insert_point(bb);

    for (int opnum = beginop; opnum < endop; ++opnum) {
        const Opcode& op        = inst()->ops()[opnum];
        const OpDescriptor* opd = shadingsys().op_descriptor(op.opname());
        if (opd && opd->llvmgenwide) {
            if (shadingsys().debug_uninit() /* debug uninitialized vals */)
                llvm_generate_debug_uninit(op);
            if (shadingsys().llvm_debug_ops())
                llvm_generate_debug_op_printf(op);
                // TODO: optionally enable
#ifdef OSL_DEV
            std::cout << "Generating :" << op.opname() << std::endl;
            if (requiresMasking(opnum))
                std::cout << " with MASKING";
            std::cout << std::endl;
#endif
            if (ll.debug_is_enabled()) {
                ll.debug_set_location(op.sourcefile(), op.sourceline() <= 0
                                                           ? 1
                                                           : op.sourceline());
            }
            {
                auto op_masking_scope = ll.create_masking_scope(
                    /*enabled=*/op.requires_masking());
                bool ok = (*opd->llvmgenwide)(*this, opnum);
                if (!ok)
                    return false;
            }
            if (shadingsys().debug_nan() /* debug NaN/Inf */
                && op.farthest_jump() < 0 /* Jumping ops don't need it */) {
                llvm_generate_debugnan(op);
            }
        } else if (op.opname() == op_nop || op.opname() == op_end) {
            // Skip this op, it does nothing...
        } else {
            shadingcontext()->errorf("LLVMOSL: Unsupported op %s in layer %s\n",
                                     op.opname(), inst()->layername());
            return false;
        }

        // If the op we coded jumps around, skip past its recursive block
        // executions.
        int next = op.farthest_jump();
        OSL_DEV_ONLY(std::cout << "farthest_jump=" << next << std::endl);
        if (next >= 0)
            opnum = next - 1;
    }
    return true;
}



llvm::Function*
BatchedBackendLLVM::build_llvm_init()
{
    // Make a group init function: void group_init(ShaderGlobals*, GroupData*)
    // Note that the GroupData* is passed as a void*.
    OSL_ASSERT(m_library_selector);
    std::string unique_name = Strutil::sprintf("%s_group_%d_init",
                                               m_library_selector,
                                               group().id());
    ll.current_function(ll.make_function(unique_name, false,
                                         ll.type_void(),  // return type
                                         llvm_type_sg_ptr(),
                                         llvm_type_groupdata_ptr(),
                                         ll.type_int()));

    if (ll.debug_is_enabled()) {
        ustring file_name
            = group()[0]->op(group()[0]->maincodebegin()).sourcefile();
        unsigned int method_line = 0;
        ll.debug_push_function(unique_name, file_name, method_line);
    }

    // Get shader globals and groupdata pointers
    m_llvm_shaderglobals_ptr = ll.current_function_arg(0);  //arg_it++;
    m_llvm_groupdata_ptr     = ll.current_function_arg(1);  //arg_it++;
    // TODO: do we need to utilize the shader mask in the init function?
    //llvm::Value * llvm_initial_shader_mask_value = ll.current_function_arg(2); //arg_it++;

    // New function, reset temp matrix pointer
    m_llvm_temp_wide_matrix_ptr             = nullptr;
    m_llvm_temp_batched_texture_options_ptr = nullptr;
    m_llvm_temp_batched_trace_options_ptr   = nullptr;

    // Set up a new IR builder
    llvm::BasicBlock* entry_bb = ll.new_basic_block(unique_name);
    ll.new_builder(entry_bb);

    ll.assume_ptr_is_aligned(m_llvm_shaderglobals_ptr, 64);
    ll.assume_ptr_is_aligned(m_llvm_groupdata_ptr, 64);

#if 0 /* helpful for debugging */
    if (llvm_debug()) {
        llvm_gen_debug_printf (Strutil::sprintf("\n\n\n\nGROUP! %s",group().name()));
        llvm_gen_debug_printf ("enter group initlayer %d %s %s");                               this->layer(), inst()->layername(), inst()->shadername()));
    }
#endif

    // Group init clears all the "layer_run" and "userdata_initialized" flags.
    if (m_num_used_layers > 1) {
        // Round up to a 64 bit boundary
        int sz = 16 * ((m_num_used_layers + 15) / 16) * sizeof(int);

        ll.op_memset(ll.void_ptr(layer_run_ref(0)), 0, sz, 4 /*align*/);
    }
    int num_userdata = (int)group().m_userdata_names.size();
    if (num_userdata) {
        int sz = num_userdata * sizeof(int);
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
        llvm_gen_debug_printf (Strutil::sprintf("exit group init %s",
                                               group().name());
#endif
    ll.op_return();

    if (llvm_debug())
        std::cout << "group init func (" << unique_name << ") "
                  << " after llvm  = "
                  << ll.bitcode_string(ll.current_function()) << "\n";

    if (ll.debug_is_enabled()) {
        ll.debug_pop_function();
    }

    ll.end_builder();  // clear the builder

    OSL_ASSERT(
        m_temp_scopes.empty()
        && "LOGIC BUG, all BatchedBackendLLVM::TempScope's should be destroyed by now");
    // Any temp allocations we've been tracking in the function's scope
    // will no longer be valid
    m_temp_allocs.clear();

    return ll.current_function();
}


llvm::Function*
BatchedBackendLLVM::build_llvm_instance(bool groupentry)
{
    // Make a layer function: void layer_func(ShaderGlobals*, GroupData*)
    // Note that the GroupData* is passed as a void*.
    OSL_ASSERT(m_library_selector);
    std::string unique_layer_name
        = Strutil::sprintf("%s_%s", m_library_selector,
                           layer_function_name().c_str());

    bool is_entry_layer = group().is_entry_layer(layer());
    ll.current_function(ll.make_function(
        unique_layer_name,
        !is_entry_layer,  // fastcall for non-entry layer functions
        ll.type_void(),   // return type
        llvm_type_sg_ptr(), llvm_type_groupdata_ptr(), ll.type_int()));

    if (ll.debug_is_enabled()) {
        ustring file_name = inst()->op(inst()->maincodebegin()).sourcefile();

        unsigned int method_line
            = inst()->op(inst()->maincodebegin()).sourceline();
        ll.debug_push_function(unique_layer_name, file_name, method_line);
    }


    // Get shader globals and groupdata pointers
    m_llvm_shaderglobals_ptr = ll.current_function_arg(0);  //arg_it++;
    m_llvm_groupdata_ptr     = ll.current_function_arg(1);  //arg_it++;
    llvm::Value* llvm_initial_shader_mask_value = ll.current_function_arg(
        2);  //arg_it++;

    // New function, reset temp matrix pointer
    m_llvm_temp_wide_matrix_ptr             = nullptr;
    m_llvm_temp_batched_texture_options_ptr = nullptr;
    m_llvm_temp_batched_trace_options_ptr   = nullptr;

    llvm::BasicBlock* entry_bb = ll.new_basic_block(unique_layer_name);
    m_exit_instance_block      = NULL;

    // Set up a new IR builder
    ll.new_builder(entry_bb);

    ll.assume_ptr_is_aligned(m_llvm_shaderglobals_ptr, 64);
    ll.assume_ptr_is_aligned(m_llvm_groupdata_ptr, 64);

    // Start with fewer data lanes active based on how full batch is.
    llvm::Value* initial_shader_mask = ll.int_as_mask(
        llvm_initial_shader_mask_value);
    ll.push_shader_instance(initial_shader_mask);
//#define __OSL_TRACE_MASKS 1
#ifdef __OSL_TRACE_MASKS
    llvm_print_mask("initial_shader_mask", initial_shader_mask);
#endif

    OSL_DEV_ONLY(std::cout << "Master Shadername = "
                           << inst()->master()->shadername() << std::endl);
    OSL_DEV_ONLY(std::cout << "Master osofilename = "
                           << inst()->master()->osofilename() << std::endl);
    OSL_DEV_ONLY(std::cout << "source of maincodebegin operation = "
                           << inst()->op(inst()->maincodebegin()).sourcefile()
                           << std::endl);


    llvm::Value* layerfield = layer_run_ref(layer_remap(layer()));

    llvm::Value* previously_executed_value = nullptr;
    if (!group().is_last_layer(layer())) {
        previously_executed_value = ll.op_load(layerfield);
    }

    if (is_entry_layer && !group().is_last_layer(layer())) {
        // For entry layers, we need an extra check to see if it already
        // ran. If it has, do an early return. Otherwise, set the 'ran' flag
        // and then run the layer.
        if (shadingsys().llvm_debug_layers())
            llvm_gen_debug_printf(
                Strutil::sprintf("checking for already-run layer %d %s %s",
                                 this->layer(), inst()->layername(),
                                 inst()->shadername()));
        llvm::Value* previously_executed = ll.int_as_mask(
            previously_executed_value);
        llvm::Value* required_lanes_executed
            = ll.op_select(initial_shader_mask, previously_executed,
                           ll.wide_constant_bool(false));
        llvm::Value* all_required_lanes_already_executed
            = ll.op_eq(initial_shader_mask, required_lanes_executed);

        llvm::BasicBlock* then_block  = ll.new_basic_block();
        llvm::BasicBlock* after_block = ll.new_basic_block();
        ll.op_branch(all_required_lanes_already_executed, then_block,
                     after_block);
        // insert point is now then_block
        // we've already executed, so return early
        if (shadingsys().llvm_debug_layers())
            llvm_gen_debug_printf(Strutil::sprintf(
                "  taking early exit, already executed layer %d %s %s",
                this->layer(), inst()->layername(), inst()->shadername()));
        ll.op_return();
        ll.set_insert_point(after_block);
    }

    if (shadingsys().llvm_debug_layers())
        llvm_gen_debug_printf(
            Strutil::sprintf("enter layer %d %s %s", this->layer(),
                             inst()->layername(), inst()->shadername()));
    // Mark this layer as executed
    if (!group().is_last_layer(layer())) {
        // Caller may only be asking for a subset of the lanes to be executed
        // We don't want to loose track of lanes we have already executed, so
        // we will OR together previously & requested executed
        llvm::Value* combined_executed
            = ll.op_or(previously_executed_value,
                       llvm_initial_shader_mask_value);
        ll.op_store(combined_executed, layerfield);
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
        // Allocate space for locals, temps, aggregate constants
        if (s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp
            || s.symtype() == SymTypeConst) {
            getOrAllocateLLVMSymbol(s);
        }
        // Set initial value for constants, closures, and strings that are
        // not parameters.
        if (s.symtype() != SymTypeParam && s.symtype() != SymTypeOutputParam
            && s.symtype() != SymTypeGlobal
            && (s.is_constant() || s.typespec().is_closure_based()
                || s.typespec().is_string_based()
                || ((s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp)
                    && shadingsys().debug_uninit())))
            llvm_assign_initial_value(s, llvm_initial_shader_mask_value);
        // If debugnan is turned on, globals check that their values are ok
        if (s.symtype() == SymTypeGlobal && shadingsys().debug_nan()) {
            TypeDesc t = s.typespec().simpletype();
            if (t.basetype
                == TypeDesc::FLOAT) {  // just check float-based types
                int ncomps = t.numelements() * t.aggregate;
                if (s.is_uniform()) {
                    llvm::Value* args[]
                        = { ll.constant(ncomps),
                            llvm_void_ptr(s),
                            ll.constant((int)s.has_derivs()),
                            sg_void_ptr(),
                            ll.constant(ustring(inst()->shadername())),
                            ll.constant(0),
                            ll.constant(s.name()),
                            ll.constant(0),
                            ll.constant(ncomps),
                            ll.constant("<none>") };
                    ll.call_function(build_name("naninf_check"), args);
                } else {
                    llvm::Value* args[]
                        = { llvm_initial_shader_mask_value,
                            ll.constant(ncomps),
                            llvm_void_ptr(s),
                            ll.constant((int)s.has_derivs()),
                            sg_void_ptr(),
                            ll.constant(ustring(inst()->shadername())),
                            ll.constant(0),
                            ll.constant(s.name()),
                            ll.constant(0),
                            ll.constant(ncomps),
                            ll.constant("<none>") };
                    ll.call_function(build_name(
                                         FuncSpec("naninf_check_offset")
                                             .arg_uniform(TypeDesc::TypeInt)
                                             .mask()),
                                     args);
                }
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
        // initializing them lazily.
        if (s.symtype() == SymTypeParam && !s.lockgeom()
            && !s.typespec().is_closure() && !s.connected()
            && !s.connected_down() && shadingsys().lazy_userdata())
            continue;
        // Set initial value for params (may contain init ops)
        llvm_assign_initial_value(s, llvm_initial_shader_mask_value);
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

    build_llvm_code(inst()->maincodebegin(), inst()->maincodeend());

    if (llvm_has_exit_instance_block()) {
        ll.op_branch(m_exit_instance_block);  // also sets insert point
    }

    // Track all symbols who needed 'partial' initialization
    std::unordered_set<Symbol*> initedsyms;

    {
        // The current mask could be altered by early returns or exit
        // But for copying output parameters to connected shaders,
        // we want to use the shader mask
        ll.push_mask(initial_shader_mask, /*negate=*/false,
                     /*absolute = */ true);
        // Need to make sure we only copy the lanes that this layer populated
        // and avoid overwriting lanes that may have previously been populated
        auto mask_copying_of_connected_symbols = ll.create_masking_scope(true);

        // Transfer all of this layer's outputs into the downstream shader's
        // inputs.
        for (int layer = this->layer() + 1; layer < group().nlayers();
             ++layer) {
            // If connection is to a node not used in the next layer
            // then it may not have been analyzed properly
            // and more importantly can be skipped
            if (m_layer_remap[layer] != -1) {
                ShaderInstance* child = group()[layer];

                for (int c = 0, Nc = child->nconnections(); c < Nc; ++c) {
                    const Connection& con(child->connection(c));
                    if (con.srclayer == this->layer()) {
                        OSL_ASSERT(
                            con.src.arrayindex == -1 && con.dst.arrayindex == -1
                            && "no support for individual array element connections");
                        // Validate unsupported connection vecSrc -> vecDst[j]
                        OSL_ASSERT(
                            (con.dst.channel == -1
                             || con.src.type.aggregate() == TypeDesc::SCALAR
                             || con.src.channel != -1)
                            && "no support for vector -> vector[i] connections");

                        Symbol* srcsym(inst()->symbol(con.src.param));
                        Symbol* dstsym(child->symbol(con.dst.param));

                        // Check remaining connections to see if any channels of this
                        // aggregate need to be initialize.
                        if (con.dst.channel != -1
                            && initedsyms.count(dstsym) == 0) {
                            initedsyms.insert(dstsym);
                            std::bitset<32> inited(
                                0);  // Only need to be 16 (matrix4)
                            assert(dstsym->typespec().aggregate()
                                   <= inited.size());
                            unsigned ninit = dstsym->typespec().aggregate() - 1;
                            for (int rc = c + 1; rc < Nc && ninit; ++rc) {
                                const Connection& next(child->connection(rc));
                                if (next.srclayer == this->layer()) {
                                    // Allow redundant/overwriting connections, i.e:
                                    // 1.  connect layer.value[i] connect layer.value[j]
                                    // 2.  connect layer.value connect layer.value
                                    if (child->symbol(next.dst.param)
                                        == dstsym) {
                                        if (next.dst.channel != -1) {
                                            assert(next.dst.channel
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
                                llvm_assign_initial_value(*dstsym,
                                                          initial_shader_mask,
                                                          true);
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
        }
        ll.pop_mask();
        // llvm_gen_debug_printf ("done copying connections");
    }

    // All done
    if (shadingsys().llvm_debug_layers())
        llvm_gen_debug_printf(
            Strutil::sprintf("exit layer %d %s %s", this->layer(),
                             inst()->layername(), inst()->shadername()));
    ll.op_return();

    if (llvm_debug())
        std::cout << "layer_func (" << unique_layer_name << ") "
                  << this->layer() << "/" << group().nlayers()
                  << " after llvm  = "
                  << ll.bitcode_string(ll.current_function()) << "\n";

    if (ll.debug_is_enabled()) {
        ll.debug_pop_function();
    }
    ll.pop_shader_instance();
    ll.end_builder();  // clear the builder

    OSL_ASSERT(
        m_temp_scopes.empty()
        && "LOGIC BUG, all BatchedBackendLLVM::TempScope's should be destroyed by now");
    // Any temp allocations we've been tracking in the function's scope
    // will no longer be valid
    m_temp_allocs.clear();

    return ll.current_function();
}



void
BatchedBackendLLVM::initialize_llvm_group()
{
    if (ll.debug_is_enabled()) {
        const char* compile_unit_name = m_group.m_name.empty()
                                            ? unknown_shader_group_name.c_str()
                                            : m_group.m_name.c_str();

        ll.debug_setup_compilation_unit(compile_unit_name);
    }

    ll.setup_optimization_passes(shadingsys().llvm_optimize(),
                                 true /*targetHost*/);

    // Clear the shaderglobals and groupdata types -- they will be
    // created on demand.
    m_llvm_type_sg                      = NULL;
    m_llvm_type_groupdata               = NULL;
    m_llvm_type_closure_component       = NULL;
    m_llvm_type_batched_texture_options = NULL;
    m_llvm_type_batched_trace_options   = NULL;

    initialize_llvm_helper_function_map();

    m_target_lib_helper->init_function_map(shadingsys());

    ll.InstallLazyFunctionCreator(helper_function_lookup);

    for (HelperFuncMap::iterator i = llvm_helper_function_map.begin(),
                                 e = llvm_helper_function_map.end();
         i != e; ++i) {
        // In case we have loaded multiple target libraries, we need to filter out
        // library functions for different isa's or vector widths
        if ((i->second.vector_width != 0 && i->second.vector_width != m_width)
            || (i->second.target_isa != TargetISA::UNKNOWN
                && i->second.target_isa != ll.target_isa())) {
            continue;  // Skip declaring functions for different vector widths and ISA targets
        }
        const std::string& funcname(i->first);
        //std::cout << "OSL Library Function Fwd:" << funcname << std::endl;
        bool varargs      = false;
        const char* types = i->second.argtypes;
        int advance;
        bool ret_is_uniform;
        TypeSpec rettype = possibly_wide_type_from_code(types, &advance,
                                                        ret_is_uniform);
        types += advance;
        std::vector<llvm::Type*> params;
        if (ret_is_uniform == false) {
            // For varying return types, we pass a pointer to the wide type as the 1st
            // parameter
            params.push_back(llvm_pass_wide_type(rettype));
        }

        while (*types) {
            bool pass_is_uniform;
            TypeSpec t = possibly_wide_type_from_code(types, &advance,
                                                      pass_is_uniform);
            if (t.simpletype().basetype == TypeDesc::UNKNOWN) {
                if (*types == '*')
                    varargs = true;
                else
                    OSL_ASSERT(0);
            } else {
                if (pass_is_uniform) {
                    params.push_back(llvm_pass_type(t));
                } else {
                    params.push_back(llvm_pass_wide_type(t));
                }
            }
            types += advance;
        }
        llvm::Function* f = nullptr;
        if (ret_is_uniform) {
            f = ll.make_function(funcname, false, llvm_type(rettype), params,
                                 varargs);
        } else {
            f = ll.make_function(funcname, false, ll.type_void(), params,
                                 varargs);
        }
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

template<int WidthT>
void
BatchedBackendLLVM::build_offsets_of_BatchedShaderGlobals(
    std::vector<unsigned int>& offset_by_index)
{
    typedef OSL::BatchedShaderGlobals<WidthT> sgBatch;
    auto uniform_offset = offsetof(sgBatch, uniform);
    auto varying_offset = offsetof(sgBatch, varying);
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, renderstate));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, tracedata));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, objdata));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, context));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, renderer));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, Ci));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, raytype));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, pad0));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, pad1));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformShaderGlobals, pad2));

    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, P));
    // Triple type in LLVM, so next 2 are included in it
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dPdx));
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dPdy));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, dPdz));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, I));
    // Triple type in LLVM, so next 2 are included in it
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dIdx));
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dIdy));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, N));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, Ng));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, u));
    // Triple type in LLVM, so next 2 are included in it
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dudx));
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dudy));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, v));
    // Triple type in LLVM, so next 2 are included in it
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dvdx));
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dvdy));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, dPdu));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, dPdv));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, time));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, dtime));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingShaderGlobals<WidthT>, dPdtime));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingShaderGlobals<WidthT>, Ps));
    // Triple type in LLVM, so next 2 are included in it
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dPsdx));
    //    offset_by_index.push_back(varying_offset + offsetof(VaryingShaderGlobals<WidthT>,dPsdy));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingShaderGlobals<WidthT>, object2common));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingShaderGlobals<WidthT>, shader2common));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingShaderGlobals<WidthT>, surfacearea));
    offset_by_index.push_back(
        varying_offset
        + offsetof(VaryingShaderGlobals<WidthT>, flipHandedness));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingShaderGlobals<WidthT>, backfacing));
}


template<int WidthT>
void
BatchedBackendLLVM::build_offsets_of_BatchedTextureOptions(
    std::vector<unsigned int>& offset_by_index)
{
    auto uniform_offset = offsetof(BatchedTextureOptions<WidthT>, uniform);
    auto varying_offset = offsetof(BatchedTextureOptions<WidthT>, varying);
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingTextureOptions<WidthT>, sblur));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingTextureOptions<WidthT>, tblur));
    offset_by_index.push_back(varying_offset
                              + offsetof(VaryingTextureOptions<WidthT>, rblur));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingTextureOptions<WidthT>, swidth));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingTextureOptions<WidthT>, twidth));
    offset_by_index.push_back(
        varying_offset + offsetof(VaryingTextureOptions<WidthT>, rwidth));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, firstchannel));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, subimage));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, subimagename));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, swrap));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, twrap));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, rwrap));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, mipmode));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, interpmode));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, anisotropic));
    offset_by_index.push_back(
        uniform_offset + offsetof(UniformTextureOptions, conservative_filter));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, fill));
    offset_by_index.push_back(uniform_offset
                              + offsetof(UniformTextureOptions, missingcolor));
    offset_by_index.push_back(
        offsetof(BatchedTextureOptions<WidthT>, private_envlayout));
}

void
BatchedBackendLLVM::run()
{
    // We choose to always run a JIT function to allow scalar default values to be
    // broadcast out to GroupData, so do not skip running if a group().does_nothing()
    // TODO: Technically we could run just 1 time, then not bother afterwards

    // At this point, we already hold the lock for this group, by virtue
    // of ShadingSystemImpl::batch_jit_group.
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
        ll.module(ll.new_module("llvm_ops"));
#else
        ll.module(ll.module_from_bitcode((char*)osl_llvm_compiled_ops_block,
                                         osl_llvm_compiled_ops_size, "llvm_ops",
                                         &err));
        if (err.length())
            shadingcontext()->errorf("ParseBitcodeFile returned '%s'\n",
                                     err.c_str());
        OSL_ASSERT(ll.module());
#endif
        // Create the ExecutionEngine
        if (!ll.make_jit_execengine(
                &err, ll.lookup_isa_by_name(shadingsys().m_llvm_jit_target),
                shadingsys().llvm_debugging_symbols(),
                shadingsys().llvm_profiling_events())) {
            shadingcontext()->errorf("Failed to create engine: %s\n",
                                     err.c_str());
            OSL_ASSERT(0);
            return;
        }

        // End of mutex lock, for the OSL_LLVM_NO_BITCODE case
    }

    m_target_lib_helper = TargetLibraryHelper::build(vector_width(),
                                                     ll.target_isa());
    OSL_ASSERT(m_target_lib_helper);
    OSL_ASSERT(m_library_selector == nullptr);
    m_library_selector = m_target_lib_helper->library_selector();

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

#if 0 && defined(OSL_DEV)
    std::cout << "llvm's data layout of GroupData" << std::endl;
    ll.dump_struct_data_layout(m_llvm_type_groupdata);

    std::cout << std::endl << std::endl << "llvm's data layout of ShaderGlobalBatch" << std::endl;
    ll.dump_struct_data_layout(m_llvm_type_sg);

#endif
    {
        std::vector<unsigned int> offset_by_index;
        switch (m_width) {
        case 8:
            build_offsets_of_BatchedShaderGlobals<8>(offset_by_index);
            break;
        case 16:
            build_offsets_of_BatchedShaderGlobals<16>(offset_by_index);
            break;
        default:
            OSL_ASSERT(
                0
                && "Unsupported width of batch.  Only widths 8 and 16 are allowed");
            break;
        };
        ll.validate_struct_data_layout(m_llvm_type_sg, offset_by_index);
    }


    std::vector<llvm::Function*> funcs(nlayers, NULL);
    for (int layer = 0; layer < nlayers; ++layer) {
        set_inst(layer);
        if (m_layer_remap[layer] != -1) {
            // If no entry points were specified, the last layer is special,
            // it's the single entry point for the whole group.
            bool is_single_entry = (layer == (nlayers - 1)
                                    && group().num_entry_layers() == 0);

            OSL_DEV_ONLY(std::cout << "build_llvm_instance for layer=" << layer
                                   << std::endl);
            funcs[layer] = build_llvm_instance(is_single_entry);
        }
    }
    // llvm::Function* entry_func = group().num_entry_layers() ? NULL : funcs[m_num_used_layers-1];
    m_stat_llvm_irgen_time += timer.lap();

    if (shadingsys().m_max_local_mem_KB
        && m_llvm_local_mem / 1024 > shadingsys().m_max_local_mem_KB) {
        shadingcontext()->errorf(
            "Shader group \"%s\" needs too much local storage: %d KB",
            group().name(), m_llvm_local_mem / 1024);
    }

    // The module contains tons of "library" functions that our generated
    // IR might call. But probably not. We don't want to incur the overhead
    // of fully compiling those, so we tell LLVM_Util to turn them into
    // non-externally-visible symbols (allowing them to be discarded if not
    // used internal to the module). We need to make exceptions for our
    // entry points, as well as for all the external functions that are
    // just declarations (not definitions) in the module (which we have
    // conveniently stashed in external_function_names).
#if 0
    std::vector<std::string> entry_function_names;
    entry_function_names.push_back (ll.func_name(init_func));
    for (int layer = 0; layer < nlayers; ++layer) {
        // set_inst (layer);
        llvm::Function* f = funcs[layer];
        if (f && group().is_entry_layer(layer))
            entry_function_names.push_back (ll.func_name(f));
    }
    ll.internalize_module_functions ("osl_", external_function_names, entry_function_names);
#else
    std::unordered_set<llvm::Function*> external_functions;
    external_functions.insert(init_func);
    for (int layer = 0; layer < nlayers; ++layer) {
        // set_inst (layer);
        llvm::Function* f = funcs[layer];
        // If we plan to call bitcode_string of a layer's function after optimization
        // it may not exist after optimization unless we treat it as external.
        if (f && (group().is_entry_layer(layer) || llvm_debug())) {
            external_functions.insert(f);
        }
    }
    ll.prune_and_internalize_module(external_functions);
#endif

    // Debug code to dump the pre-optimized bitcode to a file
    if (llvm_debug() >= 2 || shadingsys().llvm_output_bitcode()) {
        // Make a safe group name that doesn't have "/" in it! Also beware
        // filename length limits.
        std::string safegroup = Strutil::replace(group().name(), "/", ".",
                                                 true);
        if (safegroup.size() > 235)
            safegroup
                = Strutil::sprintf("TRUNC_%s_%d",
                                   safegroup.substr(safegroup.size() - 235),
                                   group().id());
        std::string name = Strutil::sprintf("%s.ll", safegroup);
        std::ofstream out(name, std::ios_base::out | std::ios_base::trunc);
        if (out.good()) {
            out << ll.bitcode_string(ll.module());
        } else {
            shadingcontext()->errorf("Could not write to '%s'", name);
        }
    }

    // Optimize the LLVM IR EVEN IF it's a do-nothing group.
    // We choose to always run a JIT function to allow scalar default values to be
    // broadcast out to GroupData, so do not skip running if a group().does_nothing()
    ll.do_optimize();

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
        for (int layer = 0; layer < nlayers; ++layer)
            if (funcs[layer])
                std::cout << "func after opt  = "
                          << ll.bitcode_string(funcs[layer]) << "\n";
#endif
        std::cout.flush();
    }

    // Debug code to dump the post-optimized bitcode to a file
    if (llvm_debug() >= 2 || shadingsys().llvm_output_bitcode()) {
        // Make a safe group name that doesn't have "/" in it! Also beware
        // filename length limits.
        std::string safegroup = Strutil::replace(group().name(), "/", ".",
                                                 true);
        if (safegroup.size() > 235)
            safegroup
                = Strutil::sprintf("TRUNC_%s_%d",
                                   safegroup.substr(safegroup.size() - 235),
                                   group().id());
        std::string name = Strutil::sprintf("%s_opt.ll", safegroup);
        std::ofstream out(name, std::ios_base::out | std::ios_base::trunc);
        if (out.good()) {
            out << ll.bitcode_string(ll.module());
        } else {
            shadingcontext()->errorf("Could not write to '%s'", name);
        }
    }

    // Force the JIT to happen now and retrieve the JITed function pointers
    // for the initialization and all public entry points.
    group().llvm_compiled_wide_init(
        (RunLLVMGroupFuncWide)ll.getPointerToFunction(init_func));
    for (int layer = 0; layer < nlayers; ++layer) {
        llvm::Function* f = funcs[layer];
        if (f && group().is_entry_layer(layer))
            group().llvm_compiled_wide_layer(
                layer, (RunLLVMGroupFuncWide)ll.getPointerToFunction(f));
    }
    if (group().num_entry_layers())
        group().llvm_compiled_wide_version(NULL);
    else
        group().llvm_compiled_wide_version(
            group().llvm_compiled_wide_layer(nlayers - 1));

    // We are destroying the entire module below, no reason to bother
    // destroying individual functions

    // Free the exec and module to reclaim all the memory.  This definitely
    // saves memory, and has almost no effect on runtime.
    ll.execengine(NULL);

    // N.B. Destroying the EE should have destroyed the module as well.
    ll.module(NULL);

    m_stat_llvm_jit_time += timer.lap();

    m_stat_total_llvm_time = timer();

    if (shadingsys().m_compile_report) {
        shadingcontext()->infof("JITed shader group %s:", group().name());
        shadingcontext()->infof(
            "    (%1.2fs = %1.2f setup, %1.2f ir, %1.2f opt, %1.2f jit; local mem %dKB)",
            m_stat_total_llvm_time, m_stat_llvm_setup_time,
            m_stat_llvm_irgen_time, m_stat_llvm_opt_time, m_stat_llvm_jit_time,
            m_llvm_local_mem / 1024);
    }
}



};  // namespace pvt
OSL_NAMESPACE_EXIT
