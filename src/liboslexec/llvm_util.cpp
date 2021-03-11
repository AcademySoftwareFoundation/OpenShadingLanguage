// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <memory>
#include <cinttypes>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/thread.h>
#include <boost/thread/tss.hpp>   /* for thread_specific_ptr */

#include <OSL/oslconfig.h>
#include <OSL/llvm_util.h>
#include <OSL/wide.h>

#if OSL_LLVM_VERSION < 70
#error "LLVM minimum version required for OSL is 7.0"
#endif

#include "llvm_passes.h"

#include <llvm/InitializePasses.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#if OSL_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsX86.h>
#endif
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/Support/TargetRegistry.h>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>


#include <llvm/Pass.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/raw_ostream.h>

// additional includes for PTX generation
#include <llvm/Transforms/Utils/SymbolRewriter.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>

#ifdef OSL_USE_OPTIX
#include <optix.h>
#endif

OSL_NAMESPACE_ENTER

namespace pvt {

typedef llvm::SectionMemoryManager LLVMMemoryManager;

typedef llvm::Error LLVMErr;


namespace {

// NOTE: This is a COPY of something internal to LLVM, but since we destroy
// our LLVMMemoryManager via global variables we can't rely on the LLVM copy
// sticking around. Because of this, the variable must be declared _before_
// jitmm_hold so that the object stays valid until after we have destroyed
// all our memory managers.
struct DefaultMMapper final : public llvm::SectionMemoryManager::MemoryMapper {
    llvm::sys::MemoryBlock
    allocateMappedMemory(llvm::SectionMemoryManager::AllocationPurpose /*Purpose*/,
                         size_t NumBytes, const llvm::sys::MemoryBlock *const NearBlock,
                         unsigned Flags, std::error_code &EC) override {
        return llvm::sys::Memory::allocateMappedMemory(NumBytes, NearBlock, Flags, EC);
    }

    std::error_code protectMappedMemory(const llvm::sys::MemoryBlock &Block,
                                        unsigned Flags) override {
        return llvm::sys::Memory::protectMappedMemory(Block, Flags);
    }

    std::error_code releaseMappedMemory(llvm::sys::MemoryBlock &M) override {
        return llvm::sys::Memory::releaseMappedMemory(M);
    }
};
static DefaultMMapper llvm_default_mapper;

static OIIO::spin_mutex llvm_global_mutex;
static bool setup_done = false;
static std::unique_ptr<std::vector<std::shared_ptr<LLVMMemoryManager> >> jitmm_hold;
static int jit_mem_hold_users = 0;

}; // end anon namespace




// ScopedJitMemoryUser will keep jitmm_hold alive until the last instance
// is gone then the it will be freed.
LLVM_Util::ScopedJitMemoryUser::ScopedJitMemoryUser()
{
    OIIO::spin_lock lock (llvm_global_mutex);
    if (jit_mem_hold_users == 0) {
        OSL_ASSERT(!jitmm_hold);
        jitmm_hold.reset(new std::vector<std::shared_ptr<LLVMMemoryManager> >());
    }
    ++jit_mem_hold_users;
}


LLVM_Util::ScopedJitMemoryUser::~ScopedJitMemoryUser()
{
    OIIO::spin_lock lock (llvm_global_mutex);
    OSL_ASSERT(jit_mem_hold_users > 0);
    --jit_mem_hold_users;
    if (jit_mem_hold_users == 0) {
        jitmm_hold.reset();
    }
}



// We hold certain things (LLVM context and custom JIT memory manager)
// per thread and retained across LLVM_Util invocations.
struct LLVM_Util::PerThreadInfo::Impl {
    Impl() {}
    ~Impl() {
        delete llvm_context;
        // N.B. Do NOT delete the jitmm -- another thread may need the
        // code! Don't worry, we stashed a pointer in jitmm_hold.
    }

    llvm::LLVMContext* llvm_context = nullptr;
    LLVMMemoryManager* llvm_jitmm = nullptr;
};



LLVM_Util::PerThreadInfo::~PerThreadInfo()
{
    // Make sure destructor to PerThreadInfoImpl is only called here
    // where we know the definition of the owned PerThreadInfoImpl;
    delete m_thread_info;
}



LLVM_Util::PerThreadInfo::Impl *
LLVM_Util::PerThreadInfo::get() const
{
    if (!m_thread_info)
        m_thread_info = new Impl();
    return m_thread_info;
}



size_t
LLVM_Util::total_jit_memory_held ()
{
    // FIXME: This can't possibly be correct. It will always return 0,
    // since jitmem is a local variable.
    size_t jitmem = 0;
    OIIO::spin_lock lock (llvm_global_mutex);
    return jitmem;
}



/// MemoryManager - Create a shell that passes on requests
/// to a real LLVMMemoryManager underneath, but can be retained after the
/// dummy is destroyed.  Also, we don't pass along any deallocations.
class LLVM_Util::MemoryManager final : public LLVMMemoryManager {
protected:
    LLVMMemoryManager *mm;  // the real one
public:

    MemoryManager(LLVMMemoryManager *realmm) : mm(realmm) {}

    void notifyObjectLoaded(llvm::ExecutionEngine *EE, const llvm::object::ObjectFile &oi) override {
        mm->notifyObjectLoaded (EE, oi);
    }

    void notifyObjectLoaded (llvm::RuntimeDyld &RTDyld, const llvm::object::ObjectFile &Obj) override {
        mm->notifyObjectLoaded(RTDyld, Obj);
    }

    void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                                uintptr_t RODataSize, uint32_t RODataAlign,
                                uintptr_t RWDataSize, uint32_t RWDataAlign) override {
        return mm->reserveAllocationSpace(CodeSize, CodeAlign, RODataSize, RODataAlign, RWDataSize, RWDataAlign);
    }

    bool needsToReserveAllocationSpace() override {
        return mm->needsToReserveAllocationSpace();
    }

    void invalidateInstructionCache() override {
        mm->invalidateInstructionCache();
    }
    
    llvm::JITSymbol findSymbol(const std::string &Name) override {
        return mm->findSymbol(Name);
    }

    uint64_t getSymbolAddressInLogicalDylib (const std::string &Name) override {
        return mm->getSymbolAddressInLogicalDylib(Name);
    }

    llvm::JITSymbol findSymbolInLogicalDylib (const std::string &Name) override {
        return mm->findSymbolInLogicalDylib(Name);
    }

    // Common
    virtual ~MemoryManager() {}

    void *getPointerToNamedFunction(const std::string &Name,
                                    bool AbortOnFailure) override {
        return mm->getPointerToNamedFunction (Name, AbortOnFailure);
    }
    uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID, llvm::StringRef SectionName) override {
        return mm->allocateCodeSection(Size, Alignment, SectionID, SectionName);
    }
    uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID, llvm::StringRef SectionName,
                                 bool IsReadOnly) override {
        return mm->allocateDataSection(Size, Alignment, SectionID,
                                       SectionName, IsReadOnly);
    }
    void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) override {
        mm->registerEHFrames (Addr, LoadAddr, Size);
    }
    void deregisterEHFrames() override {
        mm->deregisterEHFrames();
    }

    uint64_t getSymbolAddress(const std::string &Name) override {
        return mm->getSymbolAddress (Name);
    }

    bool finalizeMemory(std::string *ErrMsg) override {
        return mm->finalizeMemory (ErrMsg);
    }
};



class LLVM_Util::IRBuilder final : public llvm::IRBuilder<llvm::ConstantFolder,
                                               llvm::IRBuilderDefaultInserter> {
    typedef llvm::IRBuilder<llvm::ConstantFolder,
                            llvm::IRBuilderDefaultInserter> Base;
public:
    IRBuilder(llvm::BasicBlock *TheBB) : Base(TheBB) {}
};




struct SetCommandLineOptionsForLLVM
{
    SetCommandLineOptionsForLLVM()
    {
        // Use the command line options interface to toggle static hidden options for the llvm::LegacyPassManager
        const char *argv[] = { "SetCommandLineOptionsForLLVM",
                               "-debug-pass", "Executions" };
        llvm::cl::ParseCommandLineOptions(std::extent<decltype(argv)>::value, argv, "llvm util for Open Shading Language\n");
    }
};



LLVM_Util::LLVM_Util (const PerThreadInfo &per_thread_info,
                      int debuglevel, int vector_width)
    : m_debug(debuglevel), m_thread(NULL),
      m_llvm_context(NULL), m_llvm_module(NULL),
      m_builder(NULL), m_llvm_jitmm(NULL),
      m_current_function(NULL),
      m_llvm_module_passes(NULL), m_llvm_func_passes(NULL),
      m_llvm_exec(NULL),
      m_vector_width(vector_width),
      m_llvm_type_native_mask(nullptr),
      mVTuneNotifier(nullptr),
      m_llvm_debug_builder(nullptr),
      mDebugCU(nullptr),
      mSubTypeForInlinedFunction(nullptr),
      m_ModuleIsFinalized(false),
      m_ModuleIsPruned(false),
      m_is_masking_required(false),
      m_masked_exit_count(0)
{
    OSL_ASSERT(vector_width <= MaxSupportedSimdLaneCount);

    SetupLLVM ();
    m_thread = per_thread_info.get();
    OSL_ASSERT (m_thread);

    {
        OIIO::spin_lock lock (llvm_global_mutex);
        if (! m_thread->llvm_context) {
            m_thread->llvm_context = new llvm::LLVMContext();
            //static SetCommandLineOptionsForLLVM sSetCommandLineOptionsForLLVM;
        }

        if (! m_thread->llvm_jitmm) {
            m_thread->llvm_jitmm = new LLVMMemoryManager(&llvm_default_mapper);
            OSL_DASSERT (m_thread->llvm_jitmm);
            OSL_ASSERT (jitmm_hold &&
                "An instance of OSL::pvt::LLVM_Util::ScopedJitMemoryUser must exist with a longer lifetime than this LLVM_Util object");
            jitmm_hold->emplace_back (m_thread->llvm_jitmm);
        }
        // Hold the REAL manager and use it as an argument later
        m_llvm_jitmm = m_thread->llvm_jitmm;
    }

    OSL_ASSERT(m_thread->llvm_context);
    m_llvm_context = m_thread->llvm_context;

    // Set up aliases for types we use over and over
    m_llvm_type_float = (llvm::Type *) llvm::Type::getFloatTy (*m_llvm_context);
    m_llvm_type_double = (llvm::Type *) llvm::Type::getDoubleTy (*m_llvm_context);
    m_llvm_type_int = (llvm::Type *) llvm::Type::getInt32Ty (*m_llvm_context);
    m_llvm_type_int8 = (llvm::Type *) llvm::Type::getInt8Ty (*m_llvm_context);
    m_llvm_type_int16 = (llvm::Type *) llvm::Type::getInt16Ty (*m_llvm_context);
    if (sizeof(char *) == 4)
        m_llvm_type_addrint = (llvm::Type *) llvm::Type::getInt32Ty (*m_llvm_context);
    else
        m_llvm_type_addrint = (llvm::Type *) llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_int_ptr = (llvm::PointerType *) llvm::Type::getInt32PtrTy (*m_llvm_context);
    m_llvm_type_bool = (llvm::Type *) llvm::Type::getInt1Ty (*m_llvm_context);
    m_llvm_type_bool_ptr = (llvm::PointerType *) llvm::Type::getInt1PtrTy (*m_llvm_context);
    m_llvm_type_char = (llvm::Type *) llvm::Type::getInt8Ty (*m_llvm_context);
    m_llvm_type_longlong = (llvm::Type *) llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_void = (llvm::Type *) llvm::Type::getVoidTy (*m_llvm_context);
    m_llvm_type_char_ptr = (llvm::PointerType *) llvm::Type::getInt8PtrTy (*m_llvm_context);
    m_llvm_type_float_ptr = (llvm::PointerType *) llvm::Type::getFloatPtrTy (*m_llvm_context);
    m_llvm_type_ustring_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_char_ptr, 0);
    m_llvm_type_longlong_ptr = (llvm::PointerType *) llvm::Type::getInt64PtrTy (*m_llvm_context);
    m_llvm_type_void_ptr = m_llvm_type_char_ptr;
    m_llvm_type_double_ptr = llvm::Type::getDoublePtrTy (*m_llvm_context);

    // A triple is a struct composed of 3 floats
    std::vector<llvm::Type*> triplefields(3, m_llvm_type_float);
    m_llvm_type_triple = type_struct (triplefields, "Vec3");
    m_llvm_type_triple_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_triple, 0);

    // A matrix is a struct composed 16 floats
    std::vector<llvm::Type*> matrixfields(16, m_llvm_type_float);
    m_llvm_type_matrix = type_struct (matrixfields, "Matrix4");
    m_llvm_type_matrix_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_matrix, 0);

    // Setup up wide aliases
    // TODO:  why are there casts to the base class llvm::Type *?
    m_vector_width = OIIO::floor2(OIIO::clamp(m_vector_width, 4, 16));
    m_llvm_type_wide_float = llvm_vector_type(m_llvm_type_float, m_vector_width);
    m_llvm_type_wide_double = llvm_vector_type(m_llvm_type_double, m_vector_width);
    m_llvm_type_wide_int = llvm_vector_type(m_llvm_type_int, m_vector_width);
    m_llvm_type_wide_bool = llvm_vector_type(m_llvm_type_bool, m_vector_width);
    m_llvm_type_wide_char = llvm_vector_type(m_llvm_type_char, m_vector_width);
    m_llvm_type_wide_longlong = llvm_vector_type(m_llvm_type_longlong, m_vector_width);
    
    m_llvm_type_wide_char_ptr = llvm::PointerType::get(m_llvm_type_wide_char, 0);
    m_llvm_type_wide_ustring_ptr = llvm_vector_type(m_llvm_type_char_ptr, m_vector_width);
    m_llvm_type_wide_void_ptr = llvm_vector_type(m_llvm_type_void_ptr, m_vector_width);
    m_llvm_type_wide_int_ptr = llvm::PointerType::get(m_llvm_type_wide_int, 0);
    m_llvm_type_wide_bool_ptr = llvm::PointerType::get(m_llvm_type_wide_bool, 0);
    m_llvm_type_wide_float_ptr = llvm::PointerType::get(m_llvm_type_wide_float, 0);

    // A triple is a struct composed of 3 floats
    std::vector<llvm::Type*> triple_wide_fields(3, m_llvm_type_wide_float);
    m_llvm_type_wide_triple = type_struct (triple_wide_fields, "WideVec3");
    
    // A matrix is a struct composed 16 floats
    std::vector<llvm::Type*> matrix_wide_fields(16, m_llvm_type_wide_float);
    m_llvm_type_wide_matrix = type_struct (matrix_wide_fields, "WideMatrix4");
}



LLVM_Util::~LLVM_Util ()
{
    execengine (NULL);
    delete m_llvm_module_passes;
    delete m_llvm_func_passes;
    delete m_builder;
    delete m_llvm_debug_builder;
    module (NULL);
    // DO NOT delete m_llvm_jitmm;  // just the dummy wrapper around the real MM
}



void
LLVM_Util::SetupLLVM ()
{
    OIIO::spin_lock lock (llvm_global_mutex);
    if (setup_done)
        return;
    // Some global LLVM initialization for the first thread that
    // gets here.
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
    LLVMLinkInMCJIT();
    llvm::PassRegistry &registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(registry);
    llvm::initializeScalarOpts(registry);
    llvm::initializeIPO(registry);
    llvm::initializeAnalysis(registry);
    llvm::initializeTransformUtils(registry);
    llvm::initializeInstCombine(registry);
    llvm::initializeInstrumentation(registry);
    llvm::initializeGlobalISel(registry);
    llvm::initializeTarget(registry);
    llvm::initializeCodeGen(registry);

    // PreventBitMasksFromBeingLiveinsToBasicBlocks and PrePromoteLogicalOpsOnBitMasks are defined in llvm_passes.h
    static llvm::RegisterPass<PreventBitMasksFromBeingLiveinsToBasicBlocks<8>> sRegCustomPass0("PreventBitMasksFromBeingLiveinsToBasicBlocks<8>", "Prevent Bit Masks <8xi1> From Being Liveins To Basic Blocks Pass",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);
    static llvm::RegisterPass<PreventBitMasksFromBeingLiveinsToBasicBlocks<16>> sRegCustomPass1("PreventBitMasksFromBeingLiveinsToBasicBlocks<16>", "Prevent Bit Masks <16xi1> From Being Liveins To Basic Blocks Pass",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);
    static llvm::RegisterPass<PrePromoteLogicalOpsOnBitMasks<8>> sRegCustomPass2("PrePromoteLogicalOpsOnBitMasks<8>", "Pre-promote <8xi1> logical operations to <8xi32> to avoid default <8xi16> promotion",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);
    static llvm::RegisterPass<PrePromoteLogicalOpsOnBitMasks<16>> sRegCustomPass3("PrePromoteLogicalOpsOnBitMasks<16>", "Pre-promote <16xi1> logical operations to <16xi32> to avoid default <16xi16> promotion",
                                 false /* Only looks at CFG */,
                                 false /* Analysis Pass */);

    if (debug()) {
        for (auto t : llvm::TargetRegistry::targets())
            std::cout << "Target: '" << t.getName() << "' "
                      << t.getShortDescription() << "\n";
        std::cout << "\n";
    }

    setup_done = true;
}



llvm::Module *
LLVM_Util::new_module (const char *id)
{
    return new llvm::Module(id, context());
}



bool
LLVM_Util::debug_is_enabled() const
{
    return m_llvm_debug_builder != nullptr;
}



void
LLVM_Util::debug_setup_compilation_unit(const char * compile_unit_name)
{
    OSL_ASSERT(debug_is_enabled());
    OSL_ASSERT(mDebugCU == nullptr);

    OSL_DEV_ONLY(std::cout << "debug_setup_compilation_unit"<< std::endl);

    constexpr const char * osl_identity = "OSL_v" OSL_LIBRARY_VERSION_STRING;

    mDebugCU = m_llvm_debug_builder->createCompileUnit(
        /*llvm::dwarf::DW_LANG_C*/
        llvm::dwarf::DW_LANG_C_plus_plus,
        m_llvm_debug_builder->createFile(compile_unit_name, // filename
                "." // directory
                ),
        osl_identity, // Identify the producer of debugging information and code. Usually this is a compiler version string.
        true, // isOptimized
        "<todo>", // This string lists command line options. This string is directly embedded in debug info output which may be used by a tool analyzing generated debugging information.
        OSL_VERSION, // This indicates runtime version for languages like Objective-C
        llvm::StringRef(), // SplitName = he name of the file that we'll split debug info out into.
        llvm::DICompileUnit::DebugEmissionKind::LineTablesOnly, // DICompileUnit::DebugEmissionKind
        0, // The DWOId if this is a split skeleton compile unit.
        false, // SplitDebugInlining = Whether to emit inline debug info.
        true // DebugInfoForProfiling (default=false) = Whether to emit extra debug info for profile collection.
        );

    OSL_DEV_ONLY(std::cout << "created debug module for " << compile_unit_name << std::endl);
}



void
LLVM_Util::debug_push_function(const std::string& function_name,
                               OIIO::ustring sourcefile, int sourceline)
{
    OSL_ASSERT(debug_is_enabled());
#ifdef OSL_DEV
    std::cout << "debug_push_function function_name=" << function_name
              << " sourcefile=" << sourcefile
              << " sourceline=" << sourceline << std::endl;
#endif

    llvm::DIFile * file = getOrCreateDebugFileFor(sourcefile.string());
    const unsigned int method_scope_line = 0;

    // Rather than use dummy function parameters, we'll just reuse
    // the inlined subroutine type of void func(void).
    // TODO:  Added DIType * for BatchedShaderGlobals  And Groupdata to be
    // passed into this function so proper function type can be created.
#if 0
    llvm::DISubroutineType *subType;
    {
        llvm::SmallVector<llvm::Metadata *, 8> EltTys;
        //llvm::DIType *DblTy = KSTheDebugInfo.getDoubleTy();
        llvm::DIType *debug_double_type = m_llvm_debug_builder->createBasicType(
                "double", 64, llvm::dwarf::DW_ATE_float);
        EltTys.push_back(debug_double_type);
        EltTys.push_back(debug_double_type);

        subType = m_llvm_debug_builder->createSubroutineType(
                m_llvm_debug_builder->getOrCreateTypeArray(EltTys));
    }
#endif

    OSL_ASSERT(file);
    llvm::DISubprogram *function = m_llvm_debug_builder->createFunction(
            mDebugCU, // Scope
            function_name.c_str(),  // Name
            /*function_name.c_str()*/ llvm::StringRef(), // Linkage Name
            file, // File
            static_cast<unsigned int>(sourceline), // Line Number
            mSubTypeForInlinedFunction, // subroutine type
#if OSL_LLVM_VERSION < 80
            false, // isLocalToUnit
            true,  // isDefinition
            method_scope_line,  // Scope Line
            llvm::DINode::FlagPrototyped, // Flags
            false // isOptimized
#else
            method_scope_line,  // Scope Line
            llvm::DINode::FlagPrototyped,  // Flags
            llvm::DISubprogram::toSPFlags(false /*isLocalToUnit*/, true /*isDefinition*/, false /*isOptimized*/)
#endif
            );

    OSL_ASSERT(mLexicalBlocks.empty());
    current_function()->setSubprogram(function);
    mLexicalBlocks.push_back(function);
}



void
LLVM_Util::debug_push_inlined_function(OIIO::ustring function_name,
                                       OIIO::ustring sourcefile,
                                       int sourceline)
{
#ifdef OSL_DEV
    std::cout << "debug_push_inlined_function function_name="<< function_name
              << " sourcefile=" << sourcefile
              << " sourceline=" << sourceline << std::endl;
#endif

    OSL_ASSERT(debug_is_enabled());
    OSL_ASSERT(m_builder);
    OSL_ASSERT(m_builder->getCurrentDebugLocation().get() != NULL);
    mInliningSites.push_back(m_builder->getCurrentDebugLocation().get());

    llvm::DIFile * file = getOrCreateDebugFileFor(sourcefile.string());
    unsigned int method_scope_line = 0;

    OSL_ASSERT(getCurrentDebugScope());

    llvm::DINode::DIFlags fnFlags = (llvm::DINode::DIFlags)(llvm::DINode::FlagPrototyped | llvm::DINode::FlagNoReturn);
    llvm::DISubprogram *function = nullptr;
    function = m_llvm_debug_builder->createFunction(
        mDebugCU, // Scope
        function_name.c_str(),  // Name
        // We are inlined function so not sure supplying a linkage name
        // makes sense
        /*function_name.c_str()*/llvm::StringRef(), // Linkage Name
        file, // File
        static_cast<unsigned int>(sourceline), // Line Number
        mSubTypeForInlinedFunction, // subroutine type
#if OSL_LLVM_VERSION < 80
        true, // isLocalToUnit
        true, // isDefinition
        method_scope_line, // Scope Line
        fnFlags, // Flags
        true /*false*/ //isOptimized
#else
        method_scope_line, // Scope Line,
        fnFlags,
        llvm::DISubprogram::toSPFlags(true /*isLocalToUnit*/, true /*isDefinition*/, true /*false*/ /*isOptimized*/)
#endif
        );

    mLexicalBlocks.push_back(function);
}



void
LLVM_Util::debug_pop_inlined_function()
{
    OSL_DEV_ONLY(std::cout << "debug_pop_inlined_function"<< std::endl);
    OSL_ASSERT(debug_is_enabled());

    OSL_ASSERT(!mLexicalBlocks.empty());

    llvm::DIScope *scope = mLexicalBlocks.back();
    auto *existingLbf = llvm::dyn_cast<llvm::DILexicalBlockFile>(scope);
    if (existingLbf) {
        // Allow nesting of exactly one DILexicalBlockFile, unwrap it to a
        // function.
        scope = existingLbf->getScope();
        OSL_DEV_ONLY(std::cout << "DILexicalBlockFile popped"<< std::endl);
    }

    auto *function = llvm::dyn_cast<llvm::DISubprogram>(scope);
    OSL_ASSERT(function);
    mLexicalBlocks.pop_back();

    m_llvm_debug_builder->finalizeSubprogram(function);

    // Return debug location to where the function was inlined from.
    // Necessary to avoid unnecessarily creating DILexicalBlockFile if the
    // source file changed.
    llvm::DILocation *location_inlined_at = mInliningSites.back();
    OSL_ASSERT(location_inlined_at);
    OSL_ASSERT(m_builder);
    m_builder->SetCurrentDebugLocation(llvm::DebugLoc(location_inlined_at));
    mInliningSites.pop_back();
}



void
LLVM_Util::debug_pop_function()
{
    OSL_DEV_ONLY(std::cout << "debug_pop_function" << std::endl);
    OSL_ASSERT(debug_is_enabled());

    OSL_ASSERT(!mLexicalBlocks.empty());
    llvm::DIScope *scope = mLexicalBlocks.back();
    auto *existingLbf = llvm::dyn_cast<llvm::DILexicalBlockFile>(scope);
    if (existingLbf) {
        // Allow nesting of exactly one DILexicalBlockFile
        // Unwrap it to a function
        scope = existingLbf->getScope();
        OSL_DEV_ONLY(std::cout << "DILexicalBlockFile popped" << std::endl);
    }

    auto *function = llvm::dyn_cast<llvm::DISubprogram>(scope);
    OSL_ASSERT(function);

    mLexicalBlocks.pop_back();
    OSL_ASSERT(mLexicalBlocks.empty());

    // Make sure our current debug location isn't pointing at a subprogram
    // that has been finalized, point it back to the compilation unit
    OSL_ASSERT(m_builder);
    OSL_ASSERT(m_builder->getCurrentDebugLocation().get() != nullptr);
    m_builder->SetCurrentDebugLocation(llvm::DebugLoc::get(static_cast<unsigned int>(1),
                static_cast<unsigned int>(0), /* column?  we don't know it, may be worth tracking through osl->oso*/
                getCurrentDebugScope()));

    m_llvm_debug_builder->finalizeSubprogram(function);
}



void
LLVM_Util::debug_set_location(ustring sourcefile, int sourceline)
{
    OSL_DEV_ONLY(std::cout << "LLVM_Util::debug_set_location:" << sourcefile << "(" << sourceline << ")" << std::endl);
    OSL_ASSERT(debug_is_enabled());
    OSL_ASSERT(sourceline > 0 && "GDB doesn't like 0 because its a nonsensical as a line number");

    llvm::DIScope *sp = getCurrentDebugScope();
    llvm::DILocation *inlineSite = getCurrentInliningSite();
    OSL_ASSERT(sp != nullptr);

    // If the file changed on us (due to an #include or inlined function
    // that we missed) update the scope. As we do model inlined functions,
    // don't expect this code path to be taken unless support for the
    // functioncall_nr has been disabled.
    if (sp->getFilename().compare(llvm::StringRef(sourcefile.c_str()))) {
        llvm::DIFile * file = getOrCreateDebugFileFor(sourcefile.string());

        // Don't nest DILexicalBlockFile's (don't allow DILexicalBlockFile's
        // to be a parent to another DILexicalBlockFile's). Instead make the
        // parent of the new DILexicalBlockFile the same as the existing
        // DILexicalBlockFile's parent.
        auto *existingLbf = llvm::dyn_cast<llvm::DILexicalBlockFile>(sp);
        bool requiresNewLBF = true;
        llvm::DIScope *parentScope;
        if (existingLbf) {
            parentScope = existingLbf->getScope();
            // Only allow a single LBF, check for any logic bugs here
            OSL_ASSERT(!llvm::dyn_cast<llvm::DILexicalBlockFile>(parentScope));
            // If the parent scope has the same filename, no need to create
            // a LBF we can directly use the parentScope.
            if (!parentScope->getFilename().compare(llvm::StringRef(sourcefile.c_str()))) {
                // The parent scope has the same file name, we can just use
                // it directly.
                sp = parentScope;
                requiresNewLBF = false;
            }
        } else {
            parentScope = sp;
        }
        if (requiresNewLBF) {
            OSL_ASSERT(parentScope != nullptr);
            llvm::DILexicalBlockFile *lbf = m_llvm_debug_builder->createLexicalBlockFile(parentScope, file);
            OSL_DEV_ONLY(std::cout << "createLexicalBlockFile" << std::endl);
            sp = lbf;
        }

        // Swap out the current scope for a scope to the correct file
        mLexicalBlocks.pop_back();
        mLexicalBlocks.push_back(sp);
    }
    OSL_ASSERT(sp != NULL);


    OSL_ASSERT(m_builder);
    const llvm::DebugLoc & current_debug_location = m_builder->getCurrentDebugLocation();
    bool newDebugLocation = true;
    if (current_debug_location) {
        if (sourceline == static_cast<int>(current_debug_location.getLine()) &&
           sp == current_debug_location.getScope() &&
           inlineSite == current_debug_location.getInlinedAt () ) {
            newDebugLocation = false;
        }
    }
    if (newDebugLocation) {
        llvm::DebugLoc debug_location =
                llvm::DebugLoc::get(static_cast<unsigned int>(sourceline),
                        static_cast<unsigned int>(0), /* column?  we don't know it, may be worth tracking through osl->oso*/
                        sp,
                        inlineSite);
        m_builder->SetCurrentDebugLocation(debug_location);
    }
}



namespace { // anonymous
    inline bool error_string (llvm::Error err, std::string *str) {
        if (err) {
            if (str) {
                llvm::handleAllErrors(std::move(err),
                          [str](llvm::ErrorInfoBase &E) { *str += E.message(); });
            }
            return true;
        }
        return false;
    }
} // anonymous namespace



llvm::Module *
LLVM_Util::module_from_bitcode (const char *bitcode, size_t size,
                                const std::string &name, std::string *err)
{
    if (err)
        err->clear();

    typedef llvm::Expected<std::unique_ptr<llvm::Module> > ErrorOrModule;

    llvm::MemoryBufferRef buf =
        llvm::MemoryBufferRef(llvm::StringRef(bitcode, size), name);
#  ifdef OSL_FORCE_BITCODE_PARSE
    //
    // None of the below seems to be an issue for 3.9 and above.
    // In other JIT code I've seen a related issue, though only on OS X.
    // So if it is still is broken somewhere between 3.6 and 3.8: instead of
    // defining OSL_FORCE_BITCODE_PARSE (which is slower), you may want to
    // try prepending a "_" in two methods above:
    //   LLVM_Util::MemoryManager::getPointerToNamedFunction
    //   LLVM_Util::MemoryManager::getSymbolAddress.
    //
    // Using MCJIT should not require unconditionally parsing
    // the bitcode. But for now, when using getLazyBitcodeModule to
    // lazily deserialize the bitcode, MCJIT is unable to find the
    // called functions due to disagreement about whether a leading "_"
    // is part of the symbol name.
    ErrorOrModule ModuleOrErr = llvm::parseBitcodeFile (buf, context());
#  else
    ErrorOrModule ModuleOrErr = llvm::getLazyBitcodeModule(buf, context());
#  endif

    if (err) {
        error_string(ModuleOrErr.takeError(), err);
    }
    llvm::Module *m = ModuleOrErr ? ModuleOrErr->release() : nullptr;
# if 0
    // Debugging: print all functions in the module
    for (llvm::Module::iterator i = m->begin(); i != m->end(); ++i)
        std::cout << "  found " << i->getName().data() << "\n";
# endif
    return m;
}


void
LLVM_Util::push_function_mask(llvm::Value * startMaskValue)
{
    // As each nested function (that is inlined) will have different control flow,
    // as some lanes of nested function may return early, but that would not affect
    // the lanes of the calling function, we mush have a modified mask stack for each
    // function
    llvm::Value * alloca_for_function_mask = op_alloca(type_native_mask(), 1, "inlined_function_mask");
    m_masked_subroutine_stack.push_back(MaskedSubroutineContext{alloca_for_function_mask, /* return_count = */0 });

    op_store_mask(startMaskValue, alloca_for_function_mask);


    // Give the new function its own mask so that it may be swapped out
    // to mask out lanes that have returned early,
    // and we can just pop that mask off when the function exits
    push_mask(startMaskValue,  /*negate=*/ false, /*absolute = */ true);
}

int
LLVM_Util::masked_return_count() const
{
    return masked_function_context().return_count;
}

int
LLVM_Util::masked_exit_count() const
{
    OSL_DEV_ONLY(std::cout << "masked_exit_count = " << m_masked_exit_count << std::endl);

    return m_masked_exit_count;
}

void
LLVM_Util::pop_function_mask()
{
    pop_mask();

    OSL_ASSERT(!m_masked_subroutine_stack.empty());
    m_masked_subroutine_stack.pop_back();
}

void
LLVM_Util::push_masked_loop(llvm::Value* location_of_control_mask, llvm::Value* location_of_continue_mask)
{
    // As each nested loop will have different control flow,
    // as some lanes of nested function may 'break' out early, but that would not affect
    // the lanes outside the loop, and we could have nested loops,
    // we mush have a break count for each loop
    m_masked_loop_stack.push_back(MaskedLoopContext{location_of_control_mask, location_of_continue_mask, 0, 0});
}

bool
LLVM_Util::is_innermost_loop_masked() const
{
    if(inside_of_masked_loop()) {
        return (masked_loop_context().location_of_control_mask != nullptr);
    }
    return false;
}

int
LLVM_Util::masked_break_count() const
{
    if(inside_of_masked_loop()) {
        return masked_loop_context().break_count;
    }
    return 0;
}

int
LLVM_Util::masked_continue_count() const
{
    if(inside_of_masked_loop()) {
        return masked_loop_context().continue_count;
    }
    return 0;
}


void
LLVM_Util::pop_masked_loop()
{
    m_masked_loop_stack.pop_back();
}



void
LLVM_Util::push_shader_instance(llvm::Value * startMaskValue)
{
    push_function_mask(startMaskValue);
}


void
LLVM_Util::pop_shader_instance()
{
    m_masked_exit_count = 0;
    pop_function_mask();
}



void
LLVM_Util::new_builder (llvm::BasicBlock *block)
{
    end_builder();
    if (! block)
        block = new_basic_block ();
    m_builder = new IRBuilder (block);
    if (this->debug_is_enabled()) {
        OSL_ASSERT(getCurrentDebugScope());
        m_builder->SetCurrentDebugLocation(llvm::DebugLoc::get(static_cast<unsigned int>(1),
                static_cast<unsigned int>(0), /* column?  we don't know it, may be worth tracking through osl->oso*/
                getCurrentDebugScope()));
    }

    OSL_ASSERT(m_masked_exit_count == 0);
    OSL_ASSERT(m_masked_subroutine_stack.empty());
    OSL_ASSERT(m_mask_stack.empty());
}


/// Return the current IR builder, create a new one (for the current
/// function) if necessary.
LLVM_Util::IRBuilder &
LLVM_Util::builder () {
    if (! m_builder)
        new_builder ();
    OSL_ASSERT(m_builder);
    return *m_builder;
}


void
LLVM_Util::end_builder ()
{
    delete m_builder;
    m_builder = NULL;
}



static llvm::StringMap<bool> sCpuFeatures;

static bool populateCpuFeatures()
{
    return llvm::sys::getHostCPUFeatures(sCpuFeatures);
}


static bool initCpuFeatures()
{
    // Lazy singleton behavior, populateCpuFeatures() should
    // only get called once by 1 thread per C++ static initialization rules
    static bool is_initialized = populateCpuFeatures();
    return is_initialized;
}


// The list of cpu features should correspond to the target architecture
// or feature set that the corresponding wide library.
// So if you change the target cpu or features in liboslexec/CMakeList.txt
// for any of the wide libraries, please update here to match
static const char * target_isa_names[] = {
    "UNKNOWN", "none", "x64", "SSE4.2", "AVX", "AVX2", "AVX2_noFMA",
    "AVX512", "AVX512_noFMA", "host"
};


// clang: default
// icc: default
static const char * required_cpu_features_by_x64[] = {
    "fxsr", "mmx", "sse", "sse2", "x87"
};

// clang: -march=nehalem
// icc: -xSSE4.2
static const char * required_cpu_features_by_SSE4_2[] = {
    "cx16","fxsr","mmx","popcnt",
    // "sahf", // we shouldn't need/require this feature
    "sse","sse2","sse3","sse4.1",
    "sse4.2","ssse3", "x87"
};

// clang: -march=corei7-avx
// icc: -xAVX
static const char * required_cpu_features_by_AVX[] = {
    // "aes", // we shouldn't need/require this feature
    "avx", "cx16", "fxsr", "mmx", "pclmul", "popcnt",
    // "sahf", // we shouldn't need/require this feature
    "sse",
    "sse2", "sse3", "sse4.1", "sse4.2", "ssse3", "x87"
    // ,"xsave","xsaveopt" // Save Processor Extended States, we don't use
};

// clang: -march=core-avx2
// icc: -xCORE-AVX2
static const char * required_cpu_features_by_AVX2[] = {
    // "aes", // we shouldn't need/require this feature
    "avx", "avx2", "bmi", "bmi2", "cx16", "f16c", "fma",
    // "fsgsbase", // we shouldn't need/require this feature
    "fxsr",
    // "invpcid", // Invalidate Process-Context Identifier, we don't use
    "lzcnt", "mmx", "movbe", "pclmul", "popcnt",
    // "rdrnd", // random # don't require unless we make use of it
    // "sahf", // we shouldn't need/require this feature
    "sse",
    "sse2", "sse3", "sse4.1", "sse4.2", "ssse3", "x87"
    // ,"xsave","xsaveopt" // // Save Processor Extended States, we don't use
};

// clang: -march=core-avx2 -mno-fma
// icc: -xCORE-AVX2 -no-fma
static const char * required_cpu_features_by_AVX2_noFMA[] = {
    // "aes", // we shouldn't need/require this feature
    "avx", "avx2", "bmi", "bmi2", "cx16", "f16c",
    // "fsgsbase", // we shouldn't need/require this feature
    "fxsr",
    // "invpcid", // Invalidate Process-Context Identifier, we don't use
    "lzcnt", "mmx", "movbe", "pclmul", "popcnt",
    // "rdrnd", // random # don't require unless we make use of it
    // "sahf", // we shouldn't need/require this feature
    "sse",
    "sse2", "sse3", "sse4.1", "sse4.2", "ssse3", "x87"
    // , "xsave", "xsaveopt" // Save Processor Extended States, we don't use
};

// clang: -march=skylake-avx512
// icc: -xCORE-AVX512
static const char * required_cpu_features_by_AVX512[] = {
    // "aes", // we shouldn't need/require this feature
    "adx", "avx", "avx2", "avx512bw", "avx512cd", "avx512dq",
    "avx512f", "avx512vl", "bmi", "bmi2",
    // "clflushopt", "clwb", flushing for volatile/persistent memory we shouldn't need
    "cx16",
    "f16c", "fma",
    // "fsgsbase", // we shouldn't need/require this feature,
    "fxsr",
    // "invpcid", // Invalidate Process-Context Identifier, we don't use
    "lzcnt", "mmx", "movbe",
    //"mpx", // Memory Protection Extensions, we don't use
    "pclmul",
    // "pku"//  Memory Protection Keys we shouldn't need/require this feature,
    "popcnt",
    // "prfchw", // prefetch wide we shouldn't need/require this feature,
    // "rdrnd", "rdseed", // random # don't require unless we make use of it
    // "rtm", // transaction memory we shouldn't need/require this feature,
    // "sahf", // we shouldn't need/require this feature
    "sse", "sse2", "sse3", "sse4.1", "sse4.2", "ssse3", "x87"
    // , "xsave", "xsavec", "xsaveopt", "xsaves" // Save Processor Extended States, we don't use
};

// clang: -march=skylake-avx512 -mno-fma
// icc: -xCORE-AVX512 -no-fma
static const char * required_cpu_features_by_AVX512_noFMA[] = {
    // "aes", // we shouldn't need/require this feature
    "adx", "avx", "avx2", "avx512bw", "avx512cd", "avx512dq",
    "avx512f", "avx512vl", "bmi", "bmi2",
    // "clflushopt", "clwb", flushing for volatile/persistent memory we shouldn't need
    "cx16",
    "f16c",
    // "fsgsbase", // we shouldn't need/require this feature
    "fxsr",
    // "invpcid", // Invalidate Process-Context Identifier, we don't use
    "lzcnt", "mmx", "movbe",
    //"mpx", // Memory Protection Extensions, we don't use
    "pclmul",
    // "pku"//  Memory Protection Keys we shouldn't need/require this feature,
    "popcnt",
    // "prfchw", // prefetch wide we shouldn't need/require this feature,
    // "rdrnd", "rdseed", // random # don't require unless we make use of it
    // "rtm", // transaction memory we shouldn't need/require this feature,
    // "sahf", // we shouldn't need/require this feature
    "sse", "sse2", "sse3", "sse4.1", "sse4.2", "ssse3", "x87"
    // , "xsave", "xsavec", "xsaveopt", "xsaves" // Save Processor Extended States, we don't use
};


static cspan<const char*>
get_required_cpu_features_for(TargetISA target)
{
    switch(target) {
    case TargetISA::NONE:         return {};
    case TargetISA::x64:          return required_cpu_features_by_x64;
    case TargetISA::SSE4_2:       return required_cpu_features_by_SSE4_2;
    case TargetISA::AVX:          return required_cpu_features_by_AVX;
    case TargetISA::AVX2:         return required_cpu_features_by_AVX2;
    case TargetISA::AVX2_noFMA:   return required_cpu_features_by_AVX2_noFMA;
    case TargetISA::AVX512:       return required_cpu_features_by_AVX512;
    case TargetISA::AVX512_noFMA: return required_cpu_features_by_AVX512_noFMA;
    default:
        OSL_ASSERT(0 && "incomplete required cpu features for target are not specified");
        return {};
    }
}



/*static*/ TargetISA
LLVM_Util::lookup_isa_by_name(string_view target_name)
{
    OSL_DEV_ONLY(std::cout << "lookup_isa_by_name(" << target_name << ")" << std::endl);
    TargetISA requestedISA = TargetISA::UNKNOWN;
    if (target_name != "") {
        for (int i = static_cast<int>(TargetISA::UNKNOWN); i < static_cast<int>(TargetISA::COUNT); ++i) {
            if (OIIO::Strutil::iequals(target_name, target_isa_names[i])) {
                requestedISA = static_cast<TargetISA>(i);
                OSL_DEV_ONLY(std::cout << "REQUESTED ISA:" << target_isa_names[i] << std::endl);
                break;
            }
        }
        // NOTE: we are ignoring unrecognized target strings
    }
    return requestedISA;
}



const char*
LLVM_Util::target_isa_name(TargetISA isa)
{
    return target_isa_names[static_cast<int>(isa)];
}



bool
LLVM_Util::detect_cpu_features(TargetISA requestedISA, bool no_fma)
{
    m_target_isa = TargetISA::UNKNOWN;
    m_supports_masked_stores = false;
    m_supports_llvm_bit_masks_natively = false;
    m_supports_avx512f = false;
    m_supports_avx2 = false;
    m_supports_avx = false;

    if (! initCpuFeatures()) {
        return false;  // Could not figure it out
    }

    // Try to match features to the combination of the requested ISA and
    // what the host CPU is able to support.
    switch (requestedISA) {
    case TargetISA::UNKNOWN:
        OSL_FALLTHROUGH;
    case TargetISA::HOST:
        OSL_FALLTHROUGH;
    case TargetISA::AVX512:
        if (!no_fma) {
            if (supports_isa(TargetISA::AVX512)) {
                m_target_isa = TargetISA::AVX512;
                m_supports_masked_stores = true;
                m_supports_llvm_bit_masks_natively = true;
                m_supports_avx512f = true;
                m_supports_avx2 = true;
                m_supports_avx = true;
                break;
            }
        }
        OSL_FALLTHROUGH;
    case TargetISA::AVX512_noFMA:
        if (supports_isa(TargetISA::AVX512_noFMA)) {
            m_target_isa = TargetISA::AVX512_noFMA;
            m_supports_masked_stores = true;
            m_supports_llvm_bit_masks_natively = true;
            m_supports_avx512f = true;
            m_supports_avx2 = true;
            m_supports_avx = true;
            break;
        }
        OSL_FALLTHROUGH;
    case TargetISA::AVX2:
        if (!no_fma) {
            if (supports_isa(TargetISA::AVX2)) {
                m_target_isa = TargetISA::AVX2;
                m_supports_masked_stores = true;
                m_supports_avx2 = true;
                m_supports_avx = true;
                break;
            }
        }
        OSL_FALLTHROUGH;
    case TargetISA::AVX2_noFMA:
        if (supports_isa(TargetISA::AVX2_noFMA)) {
            m_target_isa = TargetISA::AVX2_noFMA;
            m_supports_masked_stores = true;
            m_supports_avx2 = true;
            m_supports_avx = true;
            break;
        }
        OSL_FALLTHROUGH;
    case TargetISA::AVX:
        if (supports_isa(TargetISA::AVX)) {
            m_target_isa = TargetISA::AVX;
            m_supports_avx = true;
            break;
        }
        OSL_FALLTHROUGH;
    case TargetISA::SSE4_2:
        if (supports_isa(TargetISA::SSE4_2)) {
            m_target_isa = TargetISA::SSE4_2;
            break;
        }
        OSL_FALLTHROUGH;
    case TargetISA::x64:
        if (supports_isa(TargetISA::x64)) {
            m_target_isa = TargetISA::x64;
            break;
        }
        break;
    case TargetISA::NONE:
        m_target_isa = TargetISA::NONE;
        break;
    default:
        OSL_ASSERT(0 && "Unknown TargetISA");
    }
    // std::cout << "m_supports_masked_stores = " << m_supports_masked_stores << "\n";
    // std::cout << "m_supports_llvm_bit_masks_natively = " << m_supports_llvm_bit_masks_natively << "\n";
    // std::cout << "m_supports_avx512f = " << m_supports_avx512f << "\n";
    // std::cout << "m_supports_avx2 = " << m_supports_avx2 << "\n";
    // std::cout << "m_supports_avx = " << m_supports_avx << "\n";

    return true;
}



bool
LLVM_Util::supports_isa(TargetISA target)
{
    if(!initCpuFeatures())
        return false;

#ifdef OSL_DEV
    for (auto f : sCpuFeatures)
        std::cout << "Featuremap[" << f.getKey().str() << "]=" << f.getValue() << std::endl;
#endif

    if (target <= TargetISA::UNKNOWN || target >= TargetISA::COUNT) {
        return false;
    }

    auto features = get_required_cpu_features_for(target);
    OSL_DEV_ONLY(std::cout << "Inspecting features for " << target_isa_names[static_cast<int>(target)] << std::endl);
    for (auto f : features) {
        // Bug in llvm::sys::getHostCPUFeatures does not add "x87","fxsr","mpx"
        // LLVM release 9.0+ should fix "fxsr".
        // We want to leave the features in our required_cpu_features_by_XXX
        // so we can use it to enable JIT features (even though its doubtful
        // to be useful). So we will skip testing of missing features from
        // the sCpuFeatures
        if ((strncmp(f, "x87", 3) == 0) || (strncmp(f, "mpx", 3) == 0)
#if OSL_LLVM_VERSION < 90
            || (strncmp(f, "fxsr", 4) == 0)
#endif
            ) {
            continue;
        }
        OSL_DEV_ONLY(std::cout << "Testing for cpu feature[" << i << "]:" << f << std::endl);
        if (sCpuFeatures[f] == false) {
            OSL_DEV_ONLY(std::cout << "MISSING cpu feature[" << i << "]:" << f << std::endl);
            return false;
        }
    }

    // All cpu features of the requested target are supported
    OSL_DEV_ONLY(std::cout << "All required features exist to execute code compiled for target: " << target_isa_names[static_cast<int>(target)] << std::endl);
    return true;
}



// N.B. This method is never called for PTX generation, so don't be alarmed
// if it's doing x86 specific things.
llvm::ExecutionEngine *
LLVM_Util::make_jit_execengine (std::string *err,
                                TargetISA requestedISA,
                                bool debugging_symbols,
                                bool profiling_events)
{
#if OSL_GNUC_VERSION && OSL_LLVM_VERSION < 71
    // Due to ABI breakage in LLVM 7.0.[0-1] for llvm::Optional with GCC,
    // calling any llvm API's that accept an llvm::Optional parameter will break
    // ABI causing issues.
    // https://bugs.llvm.org/show_bug.cgi?id=39427
    // Fixed in llvm 7.1.0+
    OSL_ASSERT(debugging_symbols == false && "To enable llvm debug symbols with GCC you must use LLVM 7.1.0 or higher");
#endif

    execengine (NULL);   // delete and clear any existing engine
    if (err)
        err->clear ();
    llvm::EngineBuilder engine_builder ((std::unique_ptr<llvm::Module>(module())));

    engine_builder.setEngineKind (llvm::EngineKind::JIT);
    engine_builder.setErrorStr (err);
    //engine_builder.setRelocationModel(llvm::Reloc::PIC_);
    //engine_builder.setCodeModel(llvm::CodeModel::Default);
    engine_builder.setVerifyModules(true);

    // We are actually holding a LLVMMemoryManager
    engine_builder.setMCJITMemoryManager (std::unique_ptr<llvm::RTDyldMemoryManager>
        (new MemoryManager(m_llvm_jitmm)));

    engine_builder.setOptLevel (jit_aggressive()
                                ? llvm::CodeGenOpt::Aggressive
                                : llvm::CodeGenOpt::Default);

    llvm::TargetOptions options;
    // Enables FMA's in IR generation.
    // However cpu feature set may or may not support FMA's independently
    options.AllowFPOpFusion = jit_fma() ? llvm::FPOpFusion::Fast :
                                          llvm::FPOpFusion::Standard;
    // Unfortunately enabling UnsafeFPMath allows reciprocals, which we don't want for divides
    // To match results for existing unit tests we might need to disable UnsafeFPMath
    // TODO: investigate if reciprocals can be disabled by other means.
    // Perhaps enable UnsafeFPMath, then modify creation of DIV instructions
    // to remove the arcp (allow reciprocal) flag on that instructions
    options.UnsafeFPMath = false;
    // Since there are OSL langauge functions isinf and isnan,
    // we cannot assume there will not be infs and NANs
    options.NoInfsFPMath = false;
    options.NoNaNsFPMath = false;
    // We will not be setting up any exception handling for FP math
    options.NoTrappingFPMath = true;
    // Debatable, but perhaps some tests care about the sign of +0 vs. -0
    options.NoSignedZerosFPMath = false;
    // We will NOT be changing rounding mode dynamically
    options.HonorSignDependentRoundingFPMathOption = false;

    options.NoZerosInBSS = false;
    options.GuaranteedTailCallOpt = false;
    options.StackAlignmentOverride = 0;
    options.FunctionSections = true;
    options.UseInitArray = false;
    options.FloatABIType = llvm::FloatABI::Default;
    options.RelaxELFRelocations = false;
    //options.DebuggerTuning = llvm::DebuggerKind::GDB;

    options.PrintMachineCode = dumpasm();
    engine_builder.setTargetOptions(options);

    detect_cpu_features(requestedISA, !jit_fma());

    if (initCpuFeatures()) {
        OSL_DEV_ONLY(std::cout << "Building LLVM Engine for target:" << target_isa_name(m_target_isa) << std::endl);
        std::vector<std::string> attrvec;
        auto features = get_required_cpu_features_for(m_target_isa);
        for (auto f : features) {
            OSL_DEV_ONLY(std::cout << ">>>Requesting Feature:" << f << std::endl);
            attrvec.push_back(f);
        }
        engine_builder.setMAttrs(attrvec);
    }

    m_llvm_type_native_mask = m_supports_avx512f ? m_llvm_type_wide_bool
                : llvm_vector_type(m_llvm_type_int, m_vector_width);

    m_llvm_exec = engine_builder.create();
    if (! m_llvm_exec)
        return NULL;

    //const llvm::DataLayout & data_layout = m_llvm_exec->getDataLayout();
    //OSL_DEV_ONLY(std::cout << "data_layout.getStringRepresentation()=" << data_layout.getStringRepresentation() << std::endl);

    OSL_DEV_ONLY(llvm::TargetMachine * target_machine = m_llvm_exec->getTargetMachine());
    //OSL_DEV_ONLY(std::cout << "target_machine.getTargetCPU()=" << target_machine->getTargetCPU().str() << std::endl);
    OSL_DEV_ONLY(std::cout << "target_machine.getTargetFeatureString ()=" << target_machine->getTargetFeatureString ().str() << std::endl);
    //OSL_DEV_ONLY(std::cout << "target_machine.getTargetTriple ()=" << target_machine->getTargetTriple().str() << std::endl);

    // For unknown reasons the MCJIT when constructed registers the GDB listener (which is static)
    // The following is an attempt to unregister it, and pretend it was never registered in the 1st place
    // The underlying GDBRegistrationListener is static, so we are leaking it
    m_llvm_exec->UnregisterJITEventListener(llvm::JITEventListener::createGDBRegistrationListener());

    if (debugging_symbols) {
        OSL_ASSERT(m_llvm_module != nullptr);
        OSL_DEV_ONLY(std::cout << "debugging symbols"<< std::endl);

        module()->addModuleFlag(llvm::Module::Error, "Debug Info Version",
                llvm::DEBUG_METADATA_VERSION);

        OSL_MAYBE_UNUSED unsigned int modulesDebugInfoVersion = 0;
        if (auto *Val = llvm::mdconst::dyn_extract_or_null < llvm::ConstantInt
                > (module()->getModuleFlag("Debug Info Version"))) {
            modulesDebugInfoVersion = Val->getZExtValue();
        }

        OSL_ASSERT(m_llvm_debug_builder == nullptr && "Only handle creating the debug builder once");
        m_llvm_debug_builder = new llvm::DIBuilder(*m_llvm_module);

        llvm::SmallVector<llvm::Metadata *, 8> EltTys;
        mSubTypeForInlinedFunction = m_llvm_debug_builder->createSubroutineType(
                        m_llvm_debug_builder->getOrCreateTypeArray(EltTys));

        //  OSL_DEV_ONLY(std::cout)
        //  OSL_DEV_ONLY(       << "------------------>enable_debug_info<-----------------------------module flag['Debug Info Version']= ")
        //  OSL_DEV_ONLY(       << modulesDebugInfoVersion << std::endl);

        // The underlying GDBRegistrationListener is static, so we are leaking it
        m_llvm_exec->RegisterJITEventListener(llvm::JITEventListener::createGDBRegistrationListener());
    }

    if (profiling_events) {
        // These magic lines will make it so that enough symbol information
        // is injected so that running vtune will kinda tell you which shaders
        // you're in, and sometimes which function (only for functions that don't
        // get inlined. There doesn't seem to be any perf hit from this, either
        // in code quality or JIT time. It is only enabled, however, if your copy
        // of LLVM was build with -DLLVM_USE_INTEL_JITEVENTS=ON, otherwise
        // createIntelJITEventListener() is a stub that just returns nullptr.

        // TODO:  Create better VTune listener that can handle inline fuctions
        //        https://software.intel.com/en-us/node/544211
        mVTuneNotifier = llvm::JITEventListener::createIntelJITEventListener();
        if (mVTuneNotifier != NULL) {
            m_llvm_exec->RegisterJITEventListener(mVTuneNotifier);
        }
    }

    // Force it to JIT as soon as we ask it for the code pointer,
    // don't take any chances that it might JIT lazily, since we
    // will be stealing the JIT code memory from under its nose and
    // destroying the Module & ExecutionEngine.
    m_llvm_exec->DisableLazyCompilation ();
    return m_llvm_exec;
}



namespace /*anonymous*/ {
// The return value of llvm::StructLayout::getAlignment()
// changed from an int to llvm::Align, hide with accessor function
#if OSL_LLVM_VERSION < 100
    uint64_t get_alignment(const llvm::StructLayout * layout) {
        return layout->getAlignment();
    }
#else
    uint64_t get_alignment(const llvm::StructLayout * layout) {
        return layout->getAlignment().value();
    }
#endif
} // namespace anonymous



void
LLVM_Util::dump_struct_data_layout(llvm::Type *Ty)
{
    OSL_ASSERT(Ty);
    OSL_ASSERT(Ty->isStructTy());

    llvm::StructType *structTy = static_cast<llvm::StructType *>(Ty);
    const llvm::DataLayout & data_layout = m_llvm_exec->getDataLayout();

    int number_of_elements = structTy->getNumElements();
    const llvm::StructLayout * layout = data_layout.getStructLayout (structTy);
    std::cout << "dump_struct_data_layout: getSizeInBytes(" << layout->getSizeInBytes() << ") "
        << " getAlignment(" << get_alignment(layout) << ")"
        << " hasPadding(" << layout->hasPadding() << ")" << std::endl;
    for(int index=0; index < number_of_elements; ++index) {
        llvm::Type * et = structTy->getElementType(index);
        std::cout << "   element[" << index << "] offset in bytes = " << layout->getElementOffset(index) <<
                " type is ";
        {
            llvm::raw_os_ostream os_cout(std::cout);
            et->print(os_cout);
        }
        std::cout << std::endl;
    }

}



void
LLVM_Util::validate_struct_data_layout(llvm::Type *Ty, const std::vector<unsigned int> & expected_offset_by_index)
{
    OSL_ASSERT(Ty);
    OSL_ASSERT(Ty->isStructTy());

    llvm::StructType *structTy = static_cast<llvm::StructType *>(Ty);
    const llvm::DataLayout & data_layout = m_llvm_exec->getDataLayout();

    int number_of_elements = structTy->getNumElements();

    const llvm::StructLayout * layout = data_layout.getStructLayout (structTy);
    OSL_DEV_ONLY(std::cout << "dump_struct_data_layout: getSizeInBytes(" << layout->getSizeInBytes() << ") ")
    OSL_DEV_ONLY(    << " getAlignment(" << get_alignment(layout) << ")")
    OSL_DEV_ONLY(    << " hasPadding(" << layout->hasPadding() << ")" << std::endl);

    for (int index = 0; index < number_of_elements; ++index) {
        OSL_DEV_ONLY(llvm::Type * et = structTy->getElementType(index));

        auto actual_offset = layout->getElementOffset(index);

        OSL_ASSERT(index < static_cast<int>(expected_offset_by_index.size()));
        OSL_DEV_ONLY(std::cout << "   element[" << index << "] offset in bytes = " << actual_offset << " expect offset = " << expected_offset_by_index[index] <<)
        OSL_DEV_ONLY(        " type is ");
        {
            llvm::raw_os_ostream os_cout(std::cout);
            OSL_DEV_ONLY(        et->print(os_cout));
        }
        OSL_ASSERT(expected_offset_by_index[index] == actual_offset);
        OSL_DEV_ONLY(std::cout << std::endl);
    }
    if (static_cast<int>(expected_offset_by_index.size()) != number_of_elements) {
        std::cout << "   expected " << expected_offset_by_index.size() << " members but actual member count is = " << number_of_elements << std::endl;
        OSL_ASSERT(static_cast<int>(expected_offset_by_index.size()) == number_of_elements);
    }
}



void
LLVM_Util::execengine (llvm::ExecutionEngine *exec)
{
    if (nullptr != m_llvm_exec) {
        if (nullptr != mVTuneNotifier) {
            // We explicitly remove the VTune listener, so it can't be notified of the object's release.
            // As we are holding onto the memory backing the object, this should be fine.
            // It is necessary because a profiler could try and lookup info from an object that otherwise
            // would have been unregistered.
            m_llvm_exec->UnregisterJITEventListener(mVTuneNotifier);
            delete mVTuneNotifier;
            mVTuneNotifier = nullptr;
        }

        if (debug_is_enabled()) {
            // We explicitly remove the GDB listener, so it can't be notified of the object's release.
            // As we are holding onto the memory backing the object, this should be fine.
            // It is necessary because a debugger could try and lookup info from an object that otherwise
            // would have been unregistered.

            // The GDB listener is a static object, we really aren't creating one here
            m_llvm_exec->UnregisterJITEventListener(llvm::JITEventListener::createGDBRegistrationListener());
        }
        delete m_llvm_exec;
    }
    m_llvm_exec = exec;
}



void *
LLVM_Util::getPointerToFunction (llvm::Function *func)
{
    OSL_DASSERT (func && "passed NULL to getPointerToFunction");

    if (debug_is_enabled()) {
        // We have to finalize debug info before jit happens
        m_llvm_debug_builder->finalize();
    }

    llvm::ExecutionEngine *exec = execengine();
    OSL_ASSERT(!exec->isCompilingLazily());
    if (!m_ModuleIsFinalized) {
        // Avoid lock overhead when called repeatedly
        // We don't need to finalize for each function we get
        exec->finalizeObject ();
        m_ModuleIsFinalized = true;
    }

    void *f = exec->getPointerToFunction (func);
    OSL_ASSERT (f && "could not getPointerToFunction");
    return f;
}



void
LLVM_Util::InstallLazyFunctionCreator (void* (*P)(const std::string &))
{
    llvm::ExecutionEngine *exec = execengine();
    exec->InstallLazyFunctionCreator (P);
}



void
LLVM_Util::setup_optimization_passes (int optlevel, bool target_host)
{
    OSL_DEV_ONLY(std::cout << "setup_optimization_passes " << optlevel);
    OSL_DASSERT (m_llvm_module_passes == NULL && m_llvm_func_passes == NULL);

    // Construct the per-function passes and module-wide (interprocedural
    // optimization) passes.

    m_llvm_func_passes = new llvm::legacy::FunctionPassManager(module());
    llvm::legacy::FunctionPassManager &fpm = (*m_llvm_func_passes);

    m_llvm_module_passes = new llvm::legacy::PassManager;
    llvm::legacy::PassManager &mpm = (*m_llvm_module_passes);

    llvm::TargetMachine* target_machine = nullptr;
    if (target_host) {
        target_machine = execengine()->getTargetMachine();
        llvm::Triple ModuleTriple(module()->getTargetTriple());
        // Add an appropriate TargetLibraryInfo pass for the module's triple.
        llvm::TargetLibraryInfoImpl TLII(ModuleTriple);
        mpm.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
        mpm.add(createTargetTransformInfoWrapperPass(target_machine ? target_machine->getTargetIRAnalysis()
                                                     : llvm::TargetIRAnalysis()));
        fpm.add(createTargetTransformInfoWrapperPass(
          target_machine  ? target_machine->getTargetIRAnalysis() : llvm::TargetIRAnalysis()));
    }

    // llvm_optimize 0-3 corresponds to the same set of optimizations
    // as clang: -O0, -O1, -O2, -O3
    // Tests on production shaders suggest the sweet spot between JIT time
    // and runtime performance is O1.
    //
    // Optlevels 10, 11, 12, 13 explicitly create optimization passes. They
    // are stripped down versions of clang's -O0, -O1, -O2, -O3. They try to
    // provide similar results with improved optimization time by removing
    // some expensive passes that were repeated many times and omitting
    // other passes that are not applicable or not profitable. Useful for
    // debugging, optlevel 10 adds next to no additional passes.
    switch(optlevel) {
    default:
    {
        // For LLVM 3.0 and higher, llvm_optimize 1-3 means to use the
        // same set of optimizations as clang -O1, -O2, -O3
        llvm::PassManagerBuilder builder;
        builder.OptLevel = optlevel;
        // Time spent in JIT is considerably higher if there is no inliner specified
        builder.Inliner = llvm::createFunctionInliningPass();
        builder.DisableUnrollLoops = false;
        builder.SLPVectorize = false;
        builder.LoopVectorize = false;
        builder.DisableTailCalls = false;
#if OSL_LLVM_VERSION < 90
        builder.DisableUnitAtATime = false;
#endif

        if (target_machine)
            target_machine->adjustPassManager(builder);

        builder.populateFunctionPassManager (fpm);
        builder.populateModulePassManager (mpm);
        break;
    }
    case 10:
        // truly add no optimizations
        break;
    case 11:
    {
        // The least we would want to do
        mpm.add(llvm::createFunctionInliningPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createGlobalDCEPass());
        break;
    }
    case 12:
    {
#if 0 // PRETTY_GOOD_KEEP_AS_REF
        mpm.add(llvm::createFunctionInliningPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createGlobalDCEPass());

        mpm.add(llvm::createTypeBasedAAWrapperPass());
        mpm.add(llvm::createBasicAAWrapperPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createSROAPass());
        mpm.add(llvm::createEarlyCSEPass());

        mpm.add(llvm::createReassociatePass());
        mpm.add(llvm::createConstantPropagationPass());
        mpm.add(llvm::createDeadInstEliminationPass());
        mpm.add(llvm::createCFGSimplificationPass());

        mpm.add(llvm::createPromoteMemoryToRegisterPass());
        mpm.add(llvm::createAggressiveDCEPass());

        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createDeadInstEliminationPass());

        mpm.add(llvm::createJumpThreadingPass());
        mpm.add(llvm::createSROAPass());
        mpm.add(llvm::createInstructionCombiningPass());

        // Added
        mpm.add(llvm::createDeadStoreEliminationPass());

        mpm.add(llvm::createGlobalDCEPass());
        mpm.add(llvm::createConstantMergePass());
#else
        mpm.add(llvm::createFunctionInliningPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createGlobalDCEPass());

        mpm.add(llvm::createTypeBasedAAWrapperPass());
        mpm.add(llvm::createBasicAAWrapperPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createSROAPass());
        mpm.add(llvm::createEarlyCSEPass());

        // Eliminate and remove as much as possible up front
        mpm.add(llvm::createReassociatePass());
        mpm.add(llvm::createConstantPropagationPass());
        mpm.add(llvm::createDeadInstEliminationPass());
        mpm.add(llvm::createCFGSimplificationPass());

        mpm.add(llvm::createPromoteMemoryToRegisterPass());
        mpm.add(llvm::createAggressiveDCEPass());

//        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createReassociatePass());
        // TODO: investigate if the loop optimization passes rely on metadata from clang
        // we might need to recreate that meta data in OSL's loop code to enable these passes
        mpm.add(llvm::createLoopRotatePass());
        mpm.add(llvm::createLICMPass());
        mpm.add(llvm::createLoopUnswitchPass(false));
//        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createIndVarSimplifyPass());
        // Don't think we emitted any idioms that should be converted to a loop
        // mpm.add(llvm::createLoopIdiomPass());
        mpm.add(llvm::createLoopDeletionPass());
        mpm.add(llvm::createLoopUnrollPass());
        // GVN is expensive but should pay for itself in reducing JIT time
        mpm.add(llvm::createGVNPass());


        mpm.add(llvm::createSCCPPass());
//        mpm.add(llvm::createInstructionCombiningPass());
        // JumpThreading combo had a good improvement on JIT time
        mpm.add(llvm::createJumpThreadingPass());
        // optional, didn't  seem to help more than it cost
        // mpm.add(llvm::createCorrelatedValuePropagationPass());
        mpm.add(llvm::createDeadStoreEliminationPass());
        mpm.add(llvm::createAggressiveDCEPass());
        mpm.add(llvm::createCFGSimplificationPass());
        // Place late as possible to minimize #instrs it has to process
        mpm.add(llvm::createInstructionCombiningPass());

        mpm.add(llvm::createPromoteMemoryToRegisterPass());
        mpm.add(llvm::createDeadInstEliminationPass());

        mpm.add(llvm::createGlobalDCEPass());
        mpm.add(llvm::createConstantMergePass());
#endif
        break;
    }
    case 13:
    {
        mpm.add(llvm::createGlobalDCEPass());

        mpm.add(llvm::createTypeBasedAAWrapperPass());
        mpm.add(llvm::createBasicAAWrapperPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createSROAPass());
        mpm.add(llvm::createEarlyCSEPass());
        mpm.add(llvm::createLowerExpectIntrinsicPass());

        mpm.add(llvm::createReassociatePass());
        mpm.add(llvm::createConstantPropagationPass());
        mpm.add(llvm::createDeadInstEliminationPass());
        mpm.add(llvm::createCFGSimplificationPass());

        mpm.add(llvm::createPromoteMemoryToRegisterPass());
        mpm.add(llvm::createAggressiveDCEPass());

        // The InstructionCombining is much more expensive that all the other
        // optimizations, should attempt to reduce the number of times it is
        // executed, if at all
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createDeadInstEliminationPass());

        mpm.add(llvm::createSROAPass());
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createPromoteMemoryToRegisterPass());
        mpm.add(llvm::createGlobalOptimizerPass());
        mpm.add(llvm::createReassociatePass());
        mpm.add(llvm::createIPConstantPropagationPass());

        mpm.add(llvm::createDeadArgEliminationPass());
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createPruneEHPass());
        mpm.add(llvm::createPostOrderFunctionAttrsLegacyPass());
        mpm.add(llvm::createReversePostOrderFunctionAttrsPass());
        mpm.add(llvm::createFunctionInliningPass());
        mpm.add(llvm::createConstantPropagationPass());
        mpm.add(llvm::createDeadInstEliminationPass());
        mpm.add(llvm::createCFGSimplificationPass());

        mpm.add(llvm::createArgumentPromotionPass());
        mpm.add(llvm::createAggressiveDCEPass());
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createJumpThreadingPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createSROAPass());
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createTailCallEliminationPass());

        mpm.add(llvm::createFunctionInliningPass());
        mpm.add(llvm::createConstantPropagationPass());


        mpm.add(llvm::createIPSCCPPass());
        mpm.add(llvm::createDeadArgEliminationPass());
        mpm.add(llvm::createAggressiveDCEPass());
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createCFGSimplificationPass());

        mpm.add(llvm::createFunctionInliningPass());
        mpm.add(llvm::createArgumentPromotionPass());
        mpm.add(llvm::createSROAPass());

        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createReassociatePass());
        mpm.add(llvm::createLoopRotatePass());
        mpm.add(llvm::createLICMPass());
        mpm.add(llvm::createLoopUnswitchPass(false));
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createIndVarSimplifyPass());
        mpm.add(llvm::createLoopIdiomPass());
        mpm.add(llvm::createLoopDeletionPass());
        mpm.add(llvm::createLoopUnrollPass());
        mpm.add(llvm::createGVNPass());

        mpm.add(llvm::createMemCpyOptPass());
        mpm.add(llvm::createSCCPPass());
        mpm.add(llvm::createInstructionCombiningPass());
        mpm.add(llvm::createJumpThreadingPass());
        mpm.add(llvm::createCorrelatedValuePropagationPass());
        mpm.add(llvm::createDeadStoreEliminationPass());
        mpm.add(llvm::createAggressiveDCEPass());
        mpm.add(llvm::createCFGSimplificationPass());
        mpm.add(llvm::createInstructionCombiningPass());

        mpm.add(llvm::createFunctionInliningPass());
        mpm.add(llvm::createAggressiveDCEPass());
        mpm.add(llvm::createStripDeadPrototypesPass());
        mpm.add(llvm::createGlobalDCEPass());
        mpm.add(llvm::createConstantMergePass());
        mpm.add(llvm::createVerifierPass());
        break;
    }
    }; // switch(optlevel)

    // Add some extra passes if they are needed
    if (target_host) {
        if (!m_supports_llvm_bit_masks_natively) {
            switch(m_vector_width) {
            case 16:
    #if OSL_LLVM_VERSION < 80
                mpm.add(new PrePromoteLogicalOpsOnBitMasks<16>());
    #endif
                // MUST BE THE FINAL PASS!
                mpm.add(new PreventBitMasksFromBeingLiveinsToBasicBlocks<16>());
                break;
            case 8:
    #if OSL_LLVM_VERSION < 80
                mpm.add(new PrePromoteLogicalOpsOnBitMasks<8>());
    #endif
                // MUST BE THE FINAL PASS!
                mpm.add(new PreventBitMasksFromBeingLiveinsToBasicBlocks<8>());
                break;
            case 4:
                // We don't use masking or SIMD shading for 4-wide
                break;
            default:
                std::cout << "m_vector_width = " << m_vector_width << "\n";
                OSL_ASSERT(0 && "unsupported bit mask width");
            };
        }
    }
}


void
LLVM_Util::do_optimize (std::string *out_err)
{
    OSL_ASSERT (m_llvm_module && "No module to optimize!");

#if !defined(OSL_FORCE_BITCODE_PARSE)
    LLVMErr err = m_llvm_module->materializeAll();
    if (error_string(std::move(err), out_err))
        return;
#endif

    m_llvm_func_passes->doInitialization();
    for (auto&& I : m_llvm_module->functions())
        if (!I.isDeclaration())
            m_llvm_func_passes->run(I);
    m_llvm_func_passes->doFinalization();
    m_llvm_module_passes->run (*m_llvm_module);
}



// llvm::Value::getNumUses requires that the entire module be materialized
// which defeats the purpose of the materialize & prune unneeded below we
// need to avoid getNumUses and use the materialized_* iterators to count
// uses of what is "currently" in use.
static bool anyMaterializedUses(llvm::Value &val)
{
    // NOTE: any uses from unmaterialized functions will not be included in
    // this count!
    return val.materialized_use_begin() != val.use_end();
}


#ifdef OSL_DEV
static unsigned numberOfMaterializedUses(llvm::Value &val) {
    return static_cast<unsigned>(std::distance(val.materialized_use_begin(),val.use_end()));
}
#endif


void
LLVM_Util::prune_and_internalize_module (std::unordered_set<llvm::Function*> external_functions,
                    Linkage default_linkage, std::string *out_err)
{
    // Turn tracing for pruning on locally
    #if defined(OSL_DEV)
        #define __OSL_PRUNE_ONLY(...) __VA_ARGS__
    #else
        #define __OSL_PRUNE_ONLY(...)
    #endif

    bool materialized_at_least_once;
    __OSL_PRUNE_ONLY(int materialization_pass_count = 0);
    do {
        // Materializing a function may caused other unmaterialized
        // functions to be used, so we will continue to check for used
        // functions that need to be materialized until none are.
        //
        // An alternative algorithm could scan the instructions of a
        // materialized function and recursively look for calls to
        // non-materialized functions. We think the top end of the current
        // approach to be 2-3 passes and its simple.
        materialized_at_least_once = false;
        __OSL_PRUNE_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>materialize used globals & funcs pass#: " << ++materialization_pass_count << std::endl);
        for (llvm::Function& func : *m_llvm_module) {
            if (func.isMaterializable()) {
                if (anyMaterializedUses(func)) {
                    __OSL_PRUNE_ONLY(std::cout << "materialized function "<< func.getName().data() << " for " << numberOfMaterializedUses(func) << " uses" << std::endl);
                    LLVMErr err = func.materialize();
                    if (error_string(std::move(err), out_err))
                        return;
                    materialized_at_least_once = true;
                }
            }
        }
        for (llvm::GlobalAlias& global_alias : m_llvm_module->aliases()) {
            if (global_alias.isMaterializable()) {
                if (anyMaterializedUses(global_alias)) {
                    __OSL_PRUNE_ONLY(std::cout << "materialized global alias"<< global_alias.getName().data() << " for " << numberOfMaterializedUses(global_alias) << " uses" << std::endl);
                    LLVMErr err = global_alias.materialize();
                    if (error_string(std::move(err), out_err))
                        return;
                    materialized_at_least_once = true;
                }
            }
        }
        for (llvm::GlobalVariable& global : m_llvm_module->globals()) {
            if (global.isMaterializable()) {
                if (anyMaterializedUses(global)) {
                    __OSL_PRUNE_ONLY(std::cout << "materialized global "<< global.getName().data() << " for " << numberOfMaterializedUses(global) << " uses" << std::endl);
                    LLVMErr err = global.materialize();
                    if (error_string(std::move(err), out_err))
                        return;
                    materialized_at_least_once = true;
                }
            }
        }
    } while (materialized_at_least_once);

    __OSL_PRUNE_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>After: materialize used globals & funcs<<<<<<<<<<<<<<<<<<<<<<<" << std::endl);

//
//    for (llvm::Module::iterator i = m_llvm_module->begin(); i != m_llvm_module->end(); ++i)
//    {
//        llvm::Function & func = *i;
//        if (func.isMaterializable()) {
//            auto func_name = func.getName();
//            std::cout << func_name.data() << " isMaterializable with use count " << numberOfMaterializedUses(func) << std::endl;
//        }
//    }

    std::vector<llvm::Function *> unneeded_funcs;
    std::vector<llvm::GlobalVariable *> unneeded_globals;
    std::vector<llvm::GlobalAlias *> unneeded_global_aliases;

    // NOTE: the algorithm below will drop all globals, global aliases and
    // functions not explicitly identified in the "exception" list or  by
    // llvm as having internal uses. As unneeded functions are erased, this
    // causes llvm's internal use counts to drop. During the next pass, more
    // globals and functions may become needed and be erased. NOTE: As we
    // aren't linking this module, just pulling out function pointer to
    // execute, some normal behavior is skipped.  Most notably any GLOBAL
    // CONSTRUCTORS or DESTRUCTORS that exist in the modules bit code WILL
    // NOT BE CALLED.
    //
    // We notice that the GlobalVariable llvm.global_ctors gets erased.  To
    // remedy, one could add an function (with external linkage) that uses
    // the llvm.global_ctors global variable and calls each function.  One
    // would then need to get a pointer to that function and call it,
    // presumably only once before calling any other functions out of the
    // module.  Effectively mimicking what would happen in a normal
    // binary/linker loader (the module class has a helper to do just
    // that).
    //
    // Currently the functions being compiled out of the bitcode doesn't
    // really need it, so we are choosing not to further complicate things,
    // but thought this omission should be noted.
    __OSL_PRUNE_ONLY(int remove_unused_pass_count = 0);
    for (;;) {
        __OSL_PRUNE_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>remove unused globals & funcs pass#: " << ++remove_unused_pass_count << std::endl);

        for (llvm::GlobalAlias& global_alias : m_llvm_module->aliases()) {
            if (!anyMaterializedUses(global_alias)) {
                unneeded_global_aliases.push_back(&global_alias);
            } else {
                __OSL_PRUNE_ONLY(std::cout << "keep used (" << numberOfMaterializedUses(global_alias) << ") global alias:" << global_alias.getName().data() << std::endl);
            }
        }
        for (llvm::GlobalAlias* global_alias : unneeded_global_aliases) {
            __OSL_PRUNE_ONLY(std::cout << "Erasing unneeded global alias :" << global_alias->getName().data() << std::endl);
            global_alias->eraseFromParent();
        }

        for (llvm::GlobalVariable& global : m_llvm_module->globals()) {
            // Cuda target might have included RTI globals that are not used
            // by anything in the module but Optix will expect to exist.  So
            // keep any globals whose mangled name contains the rti_internal
            // substring.
            if (!anyMaterializedUses(global) &&
                (global.getName().find("rti_internal_") == llvm::StringRef::npos)) {
                unneeded_globals.push_back(&global);
            } else {
                __OSL_PRUNE_ONLY(std::cout << "keep used (" << numberOfMaterializedUses(global) << ") global :" << global.getName().data() << std::endl);
            }
        }
        for (llvm::GlobalVariable* global : unneeded_globals) {
            __OSL_PRUNE_ONLY(std::cout << "Erasing unneeded global :" << global->getName().data() << std::endl);
            global->eraseFromParent();
        }

        for (llvm::Function & func : *m_llvm_module) {
            __OSL_PRUNE_ONLY(auto func_name = func.getName());
            if (!anyMaterializedUses(func)) {
                bool is_external = external_functions.count(&func);
                if (!is_external) {
                    unneeded_funcs.push_back(&func);
                } else {
                    __OSL_PRUNE_ONLY(std::cout << "keep external func :" << func_name.data() << std::endl);
                }
            } else {
                __OSL_PRUNE_ONLY(std::cout << "keep used (" << numberOfMaterializedUses(func) << ") func :" << func_name.data() << std::endl);
            }
        }
        for (llvm::Function* func : unneeded_funcs) {
            __OSL_PRUNE_ONLY(std::cout << "Erasing unneeded func :" << func->getName().data() << std::endl);
            func->eraseFromParent();
        }

        if (unneeded_funcs.empty() && unneeded_globals.empty() && unneeded_global_aliases.empty())
            break;
        unneeded_funcs.clear();
        unneeded_globals.clear();
        unneeded_global_aliases.clear();
    }
    __OSL_PRUNE_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>After: unused globals & funcs" << std::endl);
    __OSL_PRUNE_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>internalize non-external functions" << std::endl);
    __OSL_PRUNE_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>debug()=" << debug() << std::endl);

    llvm::GlobalValue::LinkageTypes llvm_default_linkage;
    switch (default_linkage) {
    default:
        OSL_ASSERT(0 && "Unhandled default_linkage value");
        // fallthrough so llvm_default_linkage is not uninitialized
    case Linkage::External:
        llvm_default_linkage = llvm::GlobalValue::ExternalLinkage;
        break;
    case Linkage::LinkOnceODR:
        llvm_default_linkage = llvm::GlobalValue::LinkOnceODRLinkage;
        break;
    case Linkage::Internal:
        llvm_default_linkage = llvm::GlobalValue::InternalLinkage;
        break;
    case Linkage::Private:
        llvm_default_linkage = llvm::GlobalValue::PrivateLinkage;
        break;
    };

    for (llvm::Function& func : *m_llvm_module) {
        if (func.isDeclaration())
            continue;

        bool is_external = external_functions.count(&func);
        __OSL_PRUNE_ONLY(OSL_ASSERT(is_external || anyMaterializedUses(func)));

        __OSL_PRUNE_ONLY(auto existingLinkage = func.getLinkage());
        if (is_external) {
            __OSL_PRUNE_ONLY(std::cout << "setLinkage to " << func.getName().data() << " from " << existingLinkage << " to external"<< std::endl);
            func.setLinkage (llvm::GlobalValue::ExternalLinkage);
        } else {
            __OSL_PRUNE_ONLY(std::cout << "setLinkage to " << func.getName().data() << " from " << existingLinkage << " to " << llvm_default_linkage << std::endl);
            func.setLinkage (llvm_default_linkage);
            if (default_linkage == Linkage::Private) {
                // private symbols do not participate in linkage verifier
                // could fail with "comdat global value has private
                // linkage"
                func.setName("");
                if (auto* sym_tab = func.getValueSymbolTable()) {
                    for (auto symbol = sym_tab->begin(), end_of_symbols = sym_tab->end();
                         symbol != end_of_symbols;
                         ++symbol)
                    {
                        llvm::Value *val = symbol->getValue();

                        if (!llvm::isa<llvm::GlobalValue>(val) || llvm::cast<llvm::GlobalValue>(val)->hasLocalLinkage()) {
                            if (!debug() ||
                                !val->getName().startswith("llvm.dbg")) {
                                __OSL_PRUNE_ONLY(std::cout << "remove symbol table for value:  " << val->getName().data() << std::endl);
                                // Remove from symbol table by setting name to ""
                                val->setName("");
                            }
                        }
                    }
                }
            }
        }

    }
    __OSL_PRUNE_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>After: internalize non-external functions" << std::endl);

    // At this point everything should already be materialized, but we need
    // to materialize the module itself to avoid asserts checking for the
    // module's materialization when using a DEBUG version of LLVM
    LLVMErr err = m_llvm_module->materializeAll();
    if (error_string(std::move(err), out_err))
        return;

    #undef __OSL_PRUNE_ONLY

    m_ModuleIsPruned = true;
}



// Soon to be deprecated
void
LLVM_Util::internalize_module_functions (const std::string &prefix,
                                         const std::vector<std::string> &exceptions,
                                         const std::vector<std::string> &moreexceptions)
{
    for (llvm::Function& func : module()->getFunctionList()) {
        llvm::Function *sym = &func;
        std::string symname = sym->getName().str();
        if (prefix.size() && ! OIIO::Strutil::starts_with(symname, prefix))
            continue;
        bool needed = false;
        for (size_t i = 0, e = exceptions.size(); i < e; ++i)
            if (sym->getName() == exceptions[i]) {
                needed = true;
                // std::cout << "    necessary LLVM module function "
                //           << sym->getName().str() << "\n";
                break;
            }
        for (size_t i = 0, e = moreexceptions.size(); i < e; ++i)
            if (sym->getName() == moreexceptions[i]) {
                needed = true;
                // std::cout << "    necessary LLVM module function "
                //           << sym->getName().str() << "\n";
                break;
            }
        if (!needed) {
            llvm::GlobalValue::LinkageTypes linkage = sym->getLinkage();
            // std::cout << "    unnecessary LLVM module function "
            //           << sym->getName().str() << " linkage " << int(linkage) << "\n";
            if (linkage == llvm::GlobalValue::ExternalLinkage)
                sym->setLinkage (llvm::GlobalValue::LinkOnceODRLinkage);
            // ExternalLinkage means it's potentially externally callable,
            // and so will definitely have code generated.
            // LinkOnceODRLinkage keeps one copy so it can be inlined or
            // called internally to the module, but allows it to be
            // discarded otherwise.
        }
    }
#if 0
    // I don't think we need to worry about linkage of global symbols, but
    // here is an example of how to iterate over the globals anyway.
    for (llvm::Module::global_iterator iter = module()->global_begin(); iter != module()->global_end(); iter++) {
        llvm::GlobalValue *sym = llvm::dyn_cast<llvm::GlobalValue>(iter);
        if (!sym)
            continue;
        std::string symname = sym->getName();
        if (prefix.size() && ! OIIO::Strutil::starts_with(symname, prefix))
            continue;
        bool needed = false;
        for (size_t i = 0, e = exceptions.size(); i < e; ++i)
            if (sym->getName() == exceptions[i]) {
                needed = true;
                break;
            }
        if (! needed) {
            llvm::GlobalValue::LinkageTypes linkage = sym->getLinkage();
            // std::cout << "    unnecessary LLVM global " << sym->getName().str()
            //           << " linkage " << int(linkage) << "\n";
            if (linkage == llvm::GlobalValue::ExternalLinkage)
                f->setLinkage (llvm::GlobalValue::LinkOnceODRLinkage);
        }
    }
#endif
}



llvm::Function *
LLVM_Util::make_function (const std::string &name, bool fastcall,
                          llvm::Type *rettype,
                          llvm::Type *arg1,
                          llvm::Type *arg2,
                          llvm::Type *arg3,
                          llvm::Type *arg4)
{
    std::vector<llvm::Type*> argtypes;
    if (arg1)
        argtypes.emplace_back(arg1);
    if (arg2)
        argtypes.emplace_back(arg2);
    if (arg3)
        argtypes.emplace_back(arg3);
    if (arg4)
        argtypes.emplace_back(arg4);
    return make_function (name, fastcall, rettype, argtypes, false);
}



llvm::Function *
LLVM_Util::make_function (const std::string &name, bool fastcall,
                          llvm::Type *rettype,
                          const std::vector<llvm::Type*> &params,
                          bool varargs)
{
    llvm::FunctionType *functype = type_function (rettype, params, varargs);
#if OSL_LLVM_VERSION < 90
    auto maybe_func = module()->getOrInsertFunction(name, functype);
#else
    auto maybe_func = module()->getOrInsertFunction(name, functype).getCallee();
#endif
    OSL_ASSERT (maybe_func && "getOrInsertFunction returned NULL");
    OSL_ASSERT_MSG (llvm::isa<llvm::Function>(maybe_func),
                    "Declaration for %s is wrong, LLVM had to make a cast", name.c_str());
    llvm::Function *func = llvm::cast<llvm::Function>(maybe_func);

#if OSL_LLVM_VERSION >= 80
    // We have found that when running on AVX512 hardware and targeting
    // AVX512 (and to a lesser degree targeting AVX2), performance with
    // single-point shading can suffer significantly. This is ameliorated by
    // restricting the largest vector width that it will use. It doesn't
    // seem to matter when running on AVX2 hardware. We therefore do not
    // advise choosing a wide vector width on AVX512 hardware unless you
    // are using LLVM >= 8.0.
    int vectorRegisterBitWidth = 8 * sizeof(float) * m_vector_width;
    std::string vectorRegisterBitWidthString = std::to_string(vectorRegisterBitWidth);
    func->addFnAttr ("prefer-vector-width", vectorRegisterBitWidthString);
    func->addFnAttr ("min-legal-vector-width", vectorRegisterBitWidthString);
#endif

    if (fastcall)
        func->setCallingConv(llvm::CallingConv::Fast);
    return func;
}



void
LLVM_Util::add_function_mapping (llvm::Function *func, void *addr)
{
    execengine()->addGlobalMapping (func, addr);
}



llvm::Value *
LLVM_Util::current_function_arg (int a)
{
    llvm::Function::arg_iterator arg_it = current_function()->arg_begin();
    for (int i = 0;  i < a;  ++i)
        ++arg_it;
    return &(*arg_it);
}



llvm::BasicBlock *
LLVM_Util::new_basic_block (const std::string &name)
{
    return llvm::BasicBlock::Create (context(), debug() ? name : llvm::Twine::createNull(), current_function());
}



llvm::BasicBlock *
LLVM_Util::push_function (llvm::BasicBlock *after)
{
    OSL_DEV_ONLY(std::cout << "push_function" << std::endl);

    if (! after)
        after = new_basic_block ("after_function");
    m_return_block.push_back (after);

    return after;
}



bool
LLVM_Util::inside_function() const
{
    return (false == m_return_block.empty());
}



void
LLVM_Util::pop_function ()
{
    OSL_DEV_ONLY(std::cout << "pop_function" << std::endl);

    OSL_DASSERT (! m_return_block.empty());
    builder().SetInsertPoint (m_return_block.back());
    m_return_block.pop_back ();
}


void LLVM_Util::push_masked_return_block(llvm::BasicBlock *test_return)
{
    OSL_DEV_ONLY(std::cout << "push_masked_return_block" << std::endl);

    masked_function_context().return_block_stack.push_back (test_return);
}

void LLVM_Util::pop_masked_return_block()
{
    OSL_DEV_ONLY(std::cout << "pop_masked_return_block" << std::endl);
    masked_function_context().return_block_stack.pop_back();
}

bool
LLVM_Util::has_masked_return_block() const
{
    return (! masked_function_context().return_block_stack.empty());
}

llvm::BasicBlock *
LLVM_Util::masked_return_block() const
{
    OSL_ASSERT (! masked_function_context().return_block_stack.empty());
    return masked_function_context().return_block_stack.back();
}


llvm::BasicBlock *
LLVM_Util::return_block () const
{
    OSL_DASSERT (! m_return_block.empty());
    return m_return_block.back();
}



void 
LLVM_Util::push_loop (llvm::BasicBlock *step, llvm::BasicBlock *after)
{
    m_loop_step_block.push_back (step);
    m_loop_after_block.push_back (after);
}



void 
LLVM_Util::pop_loop ()
{
    OSL_DASSERT (! m_loop_step_block.empty() && ! m_loop_after_block.empty());
    m_loop_step_block.pop_back ();
    m_loop_after_block.pop_back ();
}



llvm::BasicBlock *
LLVM_Util::loop_step_block () const
{
    OSL_DASSERT (! m_loop_step_block.empty());
    return m_loop_step_block.back();
}



llvm::BasicBlock *
LLVM_Util::loop_after_block () const
{
    OSL_DASSERT (! m_loop_after_block.empty());
    return m_loop_after_block.back();
}




llvm::Type *
LLVM_Util::type_union(const std::vector<llvm::Type *> &types)
{
    llvm::DataLayout target(module());
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
        base_type = type_void_ptr();
    else if (max_align == 4)
        base_type = type_int();
    else if (max_align == 2)
        base_type = type_int16();
    else
        base_type = (llvm::Type *) llvm::Type::getInt8Ty (context());

    size_t array_len = union_size / target.getTypeStoreSize(base_type);
    return (llvm::Type *) llvm::ArrayType::get (base_type, array_len);
}



llvm::Type *
LLVM_Util::type_struct (const std::vector<llvm::Type *> &types,
                        const std::string &name, bool is_packed)
{
    return llvm::StructType::create(context(), types, name, is_packed);
}



llvm::Type *
LLVM_Util::type_ptr (llvm::Type *type)
{
    return llvm::PointerType::get (type, 0);
}



llvm::Type *
LLVM_Util::type_array (llvm::Type *type, int n)
{
    return llvm::ArrayType::get (type, n);
}



llvm::FunctionType *
LLVM_Util::type_function (llvm::Type *rettype,
                          const std::vector<llvm::Type*> &params,
                          bool varargs)
{
    return llvm::FunctionType::get (rettype, params, varargs);
}



llvm::PointerType *
LLVM_Util::type_function_ptr (llvm::Type *rettype,
                              const std::vector<llvm::Type*> &params,
                              bool varargs)
{
    llvm::FunctionType *functype = type_function (rettype, params, varargs);
    return llvm::PointerType::getUnqual (functype);
}



std::string
LLVM_Util::llvm_typename (llvm::Type *type) const
{
    std::string s;
    llvm::raw_string_ostream stream (s);
    stream << (*type);
    return stream.str();
}



llvm::Type *
LLVM_Util::llvm_typeof (llvm::Value *val) const
{
    return val->getType();
}



std::string
LLVM_Util::llvm_typenameof (llvm::Value *val) const
{
    return llvm_typename (llvm_typeof (val));
}

llvm::Constant*
LLVM_Util::wide_constant(llvm::Constant* constant_val)
{
    return llvm::ConstantDataVector::getSplat(m_vector_width, constant_val);
}


llvm::Constant*
LLVM_Util::constant(float f)
{
    return llvm::ConstantFP::get (context(), llvm::APFloat(f));
}

llvm::Constant*
LLVM_Util::wide_constant (int width, float value)
{
    return llvm::ConstantDataVector::getSplat(width, constant(value));
}

llvm::Constant*
LLVM_Util::wide_constant (float f)
{
    return wide_constant(m_vector_width, f);
}


llvm::Constant*
LLVM_Util::constant(int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(32,i));
}


llvm::Constant*
LLVM_Util::constant8(int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(8,i));
}

llvm::Constant*
LLVM_Util::constant16(uint16_t i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(16,i));
}

llvm::Constant*
LLVM_Util::constant64(uint64_t i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(64,i));
}

llvm::Constant*
LLVM_Util::constant128(uint64_t i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(128,i));
}

llvm::Constant*
LLVM_Util::constant128(uint64_t left, uint64_t right)
{
    uint64_t bigNum[2];
    bigNum[0] = left;
    bigNum[1] = right;
    llvm::ArrayRef< uint64_t > refBigNum(&bigNum[0], 2);
    return llvm::ConstantInt::get (context(), llvm::APInt(128,refBigNum));
}


llvm::Constant*
LLVM_Util::wide_constant(int width, int value)
{
    return llvm::ConstantDataVector::getSplat(width, constant(value));
}

llvm::Constant*
LLVM_Util::wide_constant(int value)
{
    return wide_constant(m_vector_width, value);
}

llvm::Constant*
LLVM_Util::constant(size_t i)
{
    int bits = sizeof(size_t)*8;
    return llvm::ConstantInt::get (context(), llvm::APInt(bits,i));
}

llvm::Constant*
LLVM_Util::wide_constant (size_t i)
{
    int bits = sizeof(size_t)*8;
    return llvm::ConstantDataVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(bits,i)));
}

llvm::Constant *
LLVM_Util::constant_bool(bool i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(1,i));
}

llvm::Constant*
LLVM_Util::wide_constant_bool (bool i)
{
    return llvm::ConstantDataVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(1,i)));
}

llvm::Value *
LLVM_Util::constant_ptr (void *p, llvm::PointerType *type)
{
    if (! type)
        type = type_void_ptr();
    return builder().CreateIntToPtr (constant (size_t (p)), type, "const pointer");
}



llvm::Value *
LLVM_Util::constant (ustring s)
{
    // Create a const size_t with the ustring contents
    size_t bits = sizeof(size_t)*8;
    llvm::Value *str = llvm::ConstantInt::get (context(),
                               llvm::APInt(bits,size_t(s.c_str()), true));
    // Then cast the int to a char*.
    return builder().CreateIntToPtr (str, type_string(), "ustring constant");
}


llvm::Value *
LLVM_Util::wide_constant (ustring s)
{
    // Create a const size_t with the ustring contents
    size_t bits = sizeof(size_t)*8;
    llvm::Value *str = llvm::ConstantInt::get (context(),
                               llvm::APInt(bits,size_t(s.c_str()), true));
    // Then cast the int to a char*.
    llvm::Value * constant_value = builder().CreateIntToPtr (str, type_string(), "ustring constant");

    return builder().CreateVectorSplat(m_vector_width, constant_value);
}


llvm::Value * LLVM_Util::llvm_mask_to_native(llvm::Value *llvm_mask) {

    OSL_ASSERT(llvm_mask->getType() == type_wide_bool());
    if (m_supports_llvm_bit_masks_natively) {
        return llvm_mask;
    }
    llvm::Value * native_mask =  builder().CreateSExt(llvm_mask, type_wide_int());
    OSL_ASSERT(native_mask);
    OSL_ASSERT(native_mask->getType() == type_native_mask());
    return native_mask;
}

llvm::Value * LLVM_Util::native_to_llvm_mask(llvm::Value *native_mask) {
    OSL_ASSERT(native_mask->getType() == type_native_mask());

    if (m_supports_llvm_bit_masks_natively) {
        return native_mask;
    }
    // Use ICMP SLT to zero initializer to look at sign bits only
    llvm::Value * llvm_mask = op_lt(native_mask,wide_constant(0));
    // vs. truncation
    // llvm::Value * llvm_mask = builder().CreateTrunc(native_mask, type_wide_bool());

    OSL_ASSERT(llvm_mask);
    OSL_ASSERT(llvm_mask->getType() == type_wide_bool());
    return llvm_mask;
}

llvm::Value *
LLVM_Util::mask_as_int(llvm::Value *mask)
{
    OSL_ASSERT(mask->getType() == type_wide_bool());

    if (m_supports_avx512f) {

        llvm::Type * intMaskType = nullptr;
        switch(m_vector_width) {
            case 16:
                // We can just reinterpret cast a 16 bit mask to a 16 bit integer
                // and all types are happy
                intMaskType = type_int16();
                break;
            case 8:
                // We can just reinterpret cast a 8 bit mask to a 8 bit integer
                // and all types are happy
                intMaskType = type_int8();
                break;
            default:
                OSL_ASSERT(0 && "unsupported native bit mask width");
        };

        llvm::Value* result = builder().CreateBitCast (mask, intMaskType);
        return builder().CreateZExt(result, type_int());
    } else if (m_supports_avx) {
        switch(m_vector_width) {
            case 16:
            {
                // We need to do more than a simple cast to an int. Since we
                // know vectorized comparisons for AVX&AVX2 end up setting 8
                // 32 bit integers to 0xFFFFFFFF or 0x00000000, We need to
                // do more than a simple cast to an int.

                // Convert <16 x i1> -> <16 x i32> -> to <2 x< 8 x i32>>
                llvm::Value * wide_int_mask = builder().CreateSExt(mask, type_wide_int());
                auto w8_int_masks = op_split_16x(wide_int_mask);

                // Now we will use the horizontal sign extraction intrinsic
                // to build a 32 bit mask value. However the only 256bit
                // version works on floats, so we will cast from int32 to
                // float beforehand
                llvm::Type* w8_float_type = llvm_vector_type(m_llvm_type_float, 8);
                std::array<llvm::Value *,2> w8_float_masks = {{
                        builder().CreateBitCast (w8_int_masks[0], w8_float_type),
                        builder().CreateBitCast (w8_int_masks[1], w8_float_type)
                }};

                llvm::Function* func = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_avx_movmsk_ps_256);

                llvm::Value *args[1] = {
                        w8_float_masks[0]
                };
                std::array<llvm::Value *,2> int8_masks;
                int8_masks[0] = builder().CreateCall (func, makeArrayRef(args));
                args[0] = w8_float_masks[1];
                int8_masks[1] = builder().CreateCall (func, makeArrayRef(args));

                llvm::Value *upper_mask = op_shl(int8_masks[1], constant(8));
                return op_or(upper_mask, int8_masks[0]);
            }
            case 8:
            {
                // We need to do more than a simple cast to an int. Since we
                // know vectorized comparisons for AVX&AVX2 end up setting 8
                // 32 bit integers to 0xFFFFFFFF or 0x00000000, We need to
                // do more than a simple cast to an int.

                // Convert <8 x i1> -> <8 x i32>
                llvm::Value * wide_int_mask = builder().CreateSExt(mask, type_wide_int());

                // Convert <8 x i32> -> <8 x f32>
                // Now we will use the horizontal sign extraction intrinsic
                // to build a 32 bit mask value. However the only 256bit
                // version works on floats, so we will cast from int32 to
                // float beforehand
                llvm::Type* w8_float_type = llvm_vector_type(m_llvm_type_float, 8);
                llvm::Value * w8_float_mask = builder().CreateBitCast (wide_int_mask, w8_float_type);

                llvm::Function* func = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_avx_movmsk_ps_256);

                llvm::Value *args[1] = {
                        w8_float_mask
                };
                llvm::Value * int8_mask;
                int8_mask = builder().CreateCall (func, makeArrayRef(args));
                return int8_mask;
            }
            default:
            {
                OSL_ASSERT(0 && "unsupported native bit mask width");
                return mask;
            }
        };
    } else {
        switch(m_vector_width) {
            case 16:
            {
                // We need to do more than a simple cast to an int. Since we
                // know vectorized comparisons for SSE4.2 ends up setting 4
                // 32 bit integers to 0xFFFFFFFF or 0x00000000, We need to
                // do more than a simple cast to an int.

                // Convert <16 x i1> -> <16 x i32> -> to <4 x< 4 x i32>>
                llvm::Value * wide_int_mask = builder().CreateSExt(mask, type_wide_int());
                auto w4_int_masks = op_quarter_16x(wide_int_mask);

                // Now we will use the horizontal sign extraction intrinsic
                // to build a 32 bit mask value. However the only 128bit
                // version works on floats, so we will cast from int32 to
                // float beforehand
                llvm::Type* w4_float_type = llvm_vector_type(m_llvm_type_float, 4);
                std::array<llvm::Value *,4> w4_float_masks = {{
                        builder().CreateBitCast (w4_int_masks[0], w4_float_type),
                        builder().CreateBitCast (w4_int_masks[1], w4_float_type),
                        builder().CreateBitCast (w4_int_masks[2], w4_float_type),
                        builder().CreateBitCast (w4_int_masks[3], w4_float_type)
                }};

                llvm::Function* func = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_sse_movmsk_ps);

                llvm::Value *args[1] = {
                        w4_float_masks[0]
                };
                std::array<llvm::Value *,4> int4_masks;
                int4_masks[0] = builder().CreateCall (func, makeArrayRef(args));
                args[0] = w4_float_masks[1];
                int4_masks[1] = builder().CreateCall (func, makeArrayRef(args));
                args[0] = w4_float_masks[2];
                int4_masks[2] = builder().CreateCall (func, makeArrayRef(args));
                args[0] = w4_float_masks[3];
                int4_masks[3] = builder().CreateCall (func, makeArrayRef(args));

                llvm::Value *bits12_15 = op_shl(int4_masks[3], constant(12));
                llvm::Value *bits8_11 = op_shl(int4_masks[2], constant(8));
                llvm::Value *bits4_7 = op_shl(int4_masks[1], constant(4));
                return op_or(bits12_15,op_or(bits8_11, op_or(bits4_7, int4_masks[0])));
            }
            case 8:
            {
                // We need to do more than a simple cast to an int. Since we
                // know vectorized comparisons for SSE4.2 ends up setting 4
                // 32 bit integers to 0xFFFFFFFF or 0x00000000, We need to
                // do more than a simple cast to an int.

                // Convert <8 x i1> -> <8 x i32> -> to <2 x< 4 x i32>>
                llvm::Value * wide_int_mask = builder().CreateSExt(mask, type_wide_int());
                auto w4_int_masks = op_split_8x(wide_int_mask);

                // Now we will use the horizontal sign extraction intrinsic
                // to build a 32 bit mask value. However the only 128bit
                // version works on floats, so we will cast from int32 to
                // float beforehand
                llvm::Type* w4_float_type = llvm_vector_type(m_llvm_type_float, 4);
                std::array<llvm::Value *,2> w4_float_masks = {{
                        builder().CreateBitCast (w4_int_masks[0], w4_float_type),
                        builder().CreateBitCast (w4_int_masks[1], w4_float_type)
                }};

                llvm::Function* func = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_sse_movmsk_ps);

                llvm::Value *args[1] = {
                        w4_float_masks[0]
                };
                std::array<llvm::Value *,2> int4_masks;
                int4_masks[0] = builder().CreateCall (func, makeArrayRef(args));
                args[0] = w4_float_masks[1];
                int4_masks[1] = builder().CreateCall (func, makeArrayRef(args));

                llvm::Value *bits4_7 = op_shl(int4_masks[1], constant(4));
                return op_or(bits4_7, int4_masks[0]);
            }
            case 4:
            {
                // We need to do more than a simple cast to an int. Since we
                // know vectorized comparisons for SSE4.2 ends up setting 4
                // 32 bit integers to 0xFFFFFFFF or 0x00000000, We need to
                // do more than a simple cast to an int.

                // Convert <4 x i1> -> <4 x i32>
                llvm::Value * wide_int_mask = builder().CreateSExt(mask, type_wide_int());

                // Now we will use the horizontal sign extraction intrinsic
                // to build a 32 bit mask value. However the only 128bit
                // version works on floats, so we will cast from int32 to
                // float beforehand
                llvm::Type* w4_float_type = llvm_vector_type(m_llvm_type_float, 4);
                llvm::Value * w4_float_mask =
                        builder().CreateBitCast (wide_int_mask, w4_float_type);

                llvm::Function* func = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_sse_movmsk_ps);

                llvm::Value *args[1] = {
                        w4_float_mask
                };
                llvm::Value * int4_mask = builder().CreateCall (func, makeArrayRef(args));

                return int4_mask;
            }
            default:
            {
                OSL_ASSERT(0 && "unsupported native bit mask width");
                return mask;
            }
        };
    }
}



llvm::Value *
LLVM_Util::mask_as_int16(llvm::Value *mask)
{
    OSL_ASSERT(mask->getType() == type_wide_bool());
    OSL_ASSERT(m_supports_llvm_bit_masks_natively);

    return builder().CreateBitCast (mask, type_int16());
}

llvm::Value *
LLVM_Util::mask_as_int8(llvm::Value *mask)
{
    OSL_ASSERT(m_supports_llvm_bit_masks_natively);
    return builder().CreateBitCast (mask, type_int8());
}

llvm::Value *
LLVM_Util::mask4_as_int8(llvm::Value *mask)
{
    OSL_ASSERT(m_supports_llvm_bit_masks_natively);
    // combine <4xi1> mask with <4xi1> zero init to get <8xi1> and cast it
    // to i8
    llvm::Value * zero_mask4 = llvm::ConstantDataVector::getSplat(4, constant_bool(false));
    return builder().CreateBitCast (op_combine_4x_vectors(mask,zero_mask4), type_int8());
}



llvm::Value *
LLVM_Util::int_as_mask(llvm::Value *value)
{
    OSL_ASSERT(value->getType() == type_int());

    llvm::Value* result;

    if (m_supports_llvm_bit_masks_natively) {

        llvm::Type * intMaskType = nullptr;
        switch(m_vector_width) {
            case 16:
                // We can just reinterpret cast a 16 bit integer to a 16 bit mask
                // and all types are happy
                intMaskType = type_int16();
                break;
            case 8:
                // We can just reinterpret cast a 8 bit integer to a 8 bit mask
                // and all types are happy
                intMaskType = type_int8();
                break;
            default:
                OSL_ASSERT(0 && "unsupported native bit mask width");
        };
        llvm::Value* intMask = builder().CreateTrunc(value, intMaskType);

        result = builder().CreateBitCast (intMask, type_wide_bool());
    } else {
        // Since we know vectorized comparisons for AVX&AVX2 end up setting
        // 8 32 bit integers to 0xFFFFFFFF or 0x00000000, We need to do more
        // than a simple cast to an int.

        // Broadcast out the int32 mask to all data lanes
        llvm::Value * wide_int_mask = widen_value(value);

        // Create a filter for each lane to 0 out the other lane's bits
        std::vector<llvm::Constant *> lane_masks(m_vector_width);
        for (int lane_index = 0; lane_index < m_vector_width; ++lane_index) {
           lane_masks[lane_index] = llvm::ConstantInt::get(type_int(), (1<<lane_index));
        }
        llvm::Value * lane_filter = llvm::ConstantVector::get(lane_masks);

        // Bitwise AND the wide_mask and the lane filter
        llvm::Value * filtered_mask = op_and(wide_int_mask, lane_filter);

        result = op_ne(filtered_mask, wide_constant(0));
    }

    OSL_ASSERT(result->getType() == type_wide_bool());

    return result;
}



llvm::Value *
LLVM_Util::test_if_mask_is_non_zero(llvm::Value *mask)
{
    OSL_ASSERT(mask->getType() == type_wide_bool());

    llvm::Type * extended_int_vector_type;
    llvm::Type * int_reinterpret_cast_vector_type;
    llvm::Value * zeroConstant;

    switch(m_vector_width) {
    case 4:
        extended_int_vector_type = (llvm::Type *) llvm_vector_type(type_int(), m_vector_width);
        int_reinterpret_cast_vector_type = (llvm::Type *) llvm::Type::getInt128Ty (*m_llvm_context);
        zeroConstant = constant128(0);
        break;
    case 8:
        extended_int_vector_type = (llvm::Type *) llvm_vector_type(type_int(), m_vector_width);
        int_reinterpret_cast_vector_type = (llvm::Type *) llvm::IntegerType::get(*m_llvm_context,256);
        zeroConstant = llvm::ConstantInt::get (context(), llvm::APInt(256,0));
        break;
    case 16:
        // TODO:  Think better way to represent for AVX512
        // also might need something other than number of vector lanes to detect AVX512
        extended_int_vector_type = (llvm::Type *) llvm_vector_type(type_int8(), m_vector_width);
        int_reinterpret_cast_vector_type = (llvm::Type *) llvm::Type::getInt128Ty (*m_llvm_context);
        zeroConstant = constant128(0);
        break;
    default:
        OSL_ASSERT(0 && "Unhandled vector width");
        extended_int_vector_type = nullptr;
        int_reinterpret_cast_vector_type = nullptr;
        zeroConstant = nullptr;
        break;
    };

    llvm::Value * wide_int_mask = builder().CreateSExt(mask, extended_int_vector_type);
    llvm::Value * mask_as_int =  builder().CreateBitCast (wide_int_mask, int_reinterpret_cast_vector_type);
    return op_ne (mask_as_int, zeroConstant);
}



llvm::Value *
LLVM_Util::test_mask_lane(llvm::Value *mask, int lane_index)
{
    OSL_ASSERT(mask->getType() == type_wide_bool());

    return builder().CreateExtractElement (mask, lane_index);
}



llvm::Value *
LLVM_Util::op_1st_active_lane_of(llvm::Value * mask)
{
    OSL_ASSERT(mask->getType() == type_wide_bool());
    // Assumes mask is not empty

    llvm::Type * intMaskType = nullptr;
    switch(m_vector_width) {
        case 16:
            // We can just reinterpret cast a 16 bit mask to a 16 bit integer
            // and all types are happy
            intMaskType = type_int16();
            break;
        case 8:
            // We can just reinterpret cast a 8 bit mask to a 8 bit integer
            // and all types are happy
            intMaskType = type_int8();
            break;
#if 0 // WIP
        case 4:
        {
            // We can just reinterpret cast a 8 bit mask to a 8 bit integer
            // and all types are happy
            intMaskType = type_int8();

//            extended_int_vector_type = (llvm::Type *) llvm::VectorType::get(llvm::Type::getInt32Ty (*m_llvm_context), m_vector_width);
//            llvm::Value * wide_int_mask = builder().CreateSExt(mask, extended_int_vector_type);
//
//            int_reinterpret_cast_vector_type = (llvm::Type *) llvm::Type::getInt128Ty (*m_llvm_context);
//            zeroConstant = constant128(0);
//
//            llvm::Value * mask_as_int =  builder().CreateBitCast (wide_int_mask, int_reinterpret_cast_vector_type);
            break;
        }
#endif
        default:
            OSL_ASSERT(0 && "unsupported native bit mask width");
    };

    // Count trailing zeros, least significant
    llvm::Type* types[] = {
            intMaskType
    };
    llvm::Function* func_cttz = llvm::Intrinsic::getDeclaration (module(),
        llvm::Intrinsic::cttz,
        makeArrayRef(types));

    llvm::Value * int_mask = builder().CreateBitCast (mask, intMaskType);
    llvm::Value *args[2] = {
            int_mask,
            constant_bool(true)
    };

    llvm::Value * firstNonZeroIndex = builder().CreateCall (func_cttz, makeArrayRef(args));
    return firstNonZeroIndex;
}



llvm::Value *
LLVM_Util::op_lanes_that_match_masked(llvm::Value* scalar_value,
                                      llvm::Value* wide_value,
                                      llvm::Value* mask)
{
    OSL_ASSERT(scalar_value->getType()->isVectorTy() == false);
    OSL_ASSERT(wide_value->getType()->isVectorTy() == true);

    llvm::Value * uniformWideValue = widen_value(scalar_value);
    llvm::Value * lanes_matching = op_eq(uniformWideValue, wide_value);
    llvm::Value * masked_lanes_matching = op_and(lanes_matching, mask);
    return masked_lanes_matching;
}



llvm::Value *
LLVM_Util::widen_value (llvm::Value *val)
{
    return builder().CreateVectorSplat(m_vector_width, val);
}



llvm::Value *
LLVM_Util::negate_mask(llvm::Value *mask)
{
    OSL_ASSERT(mask->getType() == type_wide_bool());
    return builder().CreateNot(mask);
}



llvm::Constant*
LLVM_Util::constant(const TypeDesc &type)
{
    long long *i = (long long *)&type;
    return llvm::ConstantInt::get (context(), llvm::APInt(64,*i));
}



llvm::Constant*
LLVM_Util::void_ptr_null ()
{
    return llvm::ConstantPointerNull::get (type_void_ptr());
}



llvm::Value *
LLVM_Util::ptr_to_cast (llvm::Value* val, llvm::Type *type)
{
    return builder().CreatePointerCast(val,llvm::PointerType::get(type, 0));
}



llvm::Value *
LLVM_Util::ptr_cast (llvm::Value* val, llvm::Type *type)
{
    return builder().CreatePointerCast(val,type);
}



llvm::Value *
LLVM_Util::ptr_cast (llvm::Value* val, const TypeDesc &type)
{
    return ptr_cast (val, llvm::PointerType::get (llvm_type(type), 0));
}


llvm::Value *
LLVM_Util::wide_ptr_cast (llvm::Value* val, const TypeDesc &type)
{
    return ptr_cast (val, llvm::PointerType::get (llvm_vector_type(type), 0));
}


llvm::Value *
LLVM_Util::int_to_ptr_cast (llvm::Value* val)
{
    return builder().CreateIntToPtr(val,type_void_ptr());
}



llvm::Value *
LLVM_Util::void_ptr (llvm::Value* val)
{
    return builder().CreatePointerCast(val,type_void_ptr());
}



llvm::Type *
LLVM_Util::llvm_type (const TypeDesc &typedesc)
{
    TypeDesc t = typedesc.elementtype();
    llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = type_float();
    else if (t == TypeDesc::INT)
        lt = type_int();
    else if (t == TypeDesc::STRING)
        lt = type_string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = type_triple();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = type_matrix();
    else if (t == TypeDesc::NONE)
        lt = type_void();
    else if (t == TypeDesc::UINT8)
        lt = type_char();
    else if (t == TypeDesc::PTR)
        lt = type_void_ptr();
    else {
        OSL_ASSERT_MSG (0, "not handling type %s yet", typedesc.c_str());
    }
    if (typedesc.arraylen)
        lt = llvm::ArrayType::get (lt, typedesc.arraylen);
    OSL_DASSERT(lt);
    return lt;
}



llvm::VectorType*
LLVM_Util::llvm_vector_type(llvm::Type *elementtype, unsigned numelements)
{
#if OSL_LLVM_VERSION >= 110
    return llvm::FixedVectorType::get(elementtype, numelements);
#else
    return llvm::VectorType::get(elementtype, numelements);
#endif
}



llvm::Type *
LLVM_Util::llvm_vector_type (const TypeDesc &typedesc)
{
    TypeDesc t = typedesc.elementtype();
    llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = type_wide_float();
    else if (t == TypeDesc::INT)
        lt = type_wide_int();
    else if (t == TypeDesc::STRING)
        lt = type_wide_string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = type_wide_triple();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = type_wide_matrix();
    // TODO:  No such thing as a wide void?
    // so let this fall through to error below
    // see if we ever run into it
//    else if (t == TypeDesc::NONE)
//        lt = type_wide_void();
    else if (t == TypeDesc::UINT8)
        lt = type_wide_char();
    else if (t == TypeDesc::PTR)
        lt = type_wide_void_ptr();
    else {
        std::cerr << "Bad llvm_vector_type(" << typedesc << ")\n";
        OSL_ASSERT (0 && "not handling this type yet");
    }
    if (typedesc.arraylen)
        lt = llvm::ArrayType::get (lt, typedesc.arraylen);
    OSL_DASSERT (lt);
    return lt;
}



llvm::Value *
LLVM_Util::offset_ptr (llvm::Value *ptr, int offset, llvm::Type *ptrtype)
{
    llvm::Value *i = builder().CreatePtrToInt (ptr, type_addrint());
    i = builder().CreateAdd (i, constant ((size_t)offset));
    ptr = builder().CreateIntToPtr (i, type_void_ptr());
    if (ptrtype)
        ptr = ptr_cast (ptr, ptrtype);
    return ptr;
}



void
LLVM_Util::assume_ptr_is_aligned(llvm::Value* ptr, unsigned alignment)
{
    const llvm::DataLayout& data_layout = m_llvm_exec->getDataLayout();
    builder().CreateAlignmentAssumption(data_layout, ptr, alignment);
}



llvm::Value *
LLVM_Util::op_alloca (llvm::Type *llvmtype, int n, const std::string &name, int align)
{
    // We must avoid emitting any alloca's inside loops and we wish to reuse
    // temporaries across the body of a function, which means we should not
    // emit them in conditional branches either. So always place alloca's at
    // the very beginning of a function. To do that we save the current
    // insertion point, change it to the beginning of the function, emit the
    // alloca, then restore the insertion point to where it was previously.
    auto previousIP = m_builder->saveIP();

    llvm::BasicBlock * entry_block = &current_function()->getEntryBlock();
    m_builder->SetInsertPoint(entry_block, entry_block->begin());

    llvm::ConstantInt* numalloc = (llvm::ConstantInt*)constant(n);
    llvm::AllocaInst* allocainst = builder().CreateAlloca (llvmtype, numalloc,
                                    debug() ? name : llvm::Twine::createNull());
    if (align > 0) {
#if OSL_LLVM_VERSION >= 110
        using AlignmentType = llvm::Align;
#elif OSL_LLVM_VERSION >= 100
        using AlignmentType = llvm::MaybeAlign;
#else
        using AlignmentType = int;
#endif
        allocainst->setAlignment (AlignmentType(align));
    }
    OSL_ASSERT(previousIP.isSet());
    m_builder->restoreIP(previousIP);

    return allocainst;
}



llvm::Value *
LLVM_Util::op_alloca (const TypeDesc &type, int n, const std::string &name, int align)
{
    return op_alloca (llvm_type(type.elementtype()), n*type.numelements(), name, align);
}


llvm::Value *
LLVM_Util::wide_op_alloca (const TypeDesc &type, int n, const std::string &name, int align)
{
    return op_alloca (llvm_vector_type(type.elementtype()), n*type.numelements(), name, align);
}



llvm::Value *
LLVM_Util::call_function (llvm::Value *func, cspan<llvm::Value *> args)
{
    OSL_DASSERT(func);
#if 0
    llvm::outs() << "llvm_call_function " << *func << "\n";
    llvm::outs() << nargs << " args:\n";
    for (int i = 0, nargs = args.size();  i < nargs;  ++i)
        llvm::outs() << "\t" << *(args[i]) << "\n";
#endif
    //llvm_gen_debug_printf (std::string("start ") + std::string(name));
#if OSL_LLVM_VERSION >= 110
    OSL_DASSERT(llvm::isa<llvm::Function>(func));
    llvm::Value *r = builder().CreateCall(llvm::cast<llvm::Function>(func), llvm::ArrayRef<llvm::Value *>(args.data(), args.size()));
#else
    llvm::Value *r = builder().CreateCall (func, llvm::ArrayRef<llvm::Value *>(args.data(), args.size()));
#endif
    //llvm_gen_debug_printf (std::string(" end  ") + std::string(name));
    return r;
}



llvm::Value *
LLVM_Util::call_function (const char *name, cspan<llvm::Value *> args)
{
    llvm::Function *func = module()->getFunction (name);
    return call_function (func, args);
}



void
LLVM_Util::mark_fast_func_call (llvm::Value *funccall)
{
    llvm::CallInst* call_inst = llvm::cast<llvm::CallInst>(funccall);
    call_inst->setCallingConv (llvm::CallingConv::Fast);
}



void
LLVM_Util::op_branch (llvm::BasicBlock *block)
{
    builder().CreateBr (block);
    set_insert_point (block);
}



void
LLVM_Util::op_branch (llvm::Value *cond, llvm::BasicBlock *trueblock,
                      llvm::BasicBlock *falseblock)
{
    builder().CreateCondBr (cond, trueblock, falseblock);
    set_insert_point (trueblock);
}



void
LLVM_Util::set_insert_point (llvm::BasicBlock *block)
{
    builder().SetInsertPoint (block);
}



void
LLVM_Util::op_return (llvm::Value *retval)
{
    if (retval)
        builder().CreateRet (retval);
    else
        builder().CreateRetVoid ();
}



void
LLVM_Util::op_memset (llvm::Value *ptr, int val, int len, int align)
{
    builder().CreateMemSet (ptr, builder().getInt8((unsigned char)val), uint64_t(len),
#if OSL_LLVM_VERSION >= 100
        llvm::MaybeAlign(align));
#else
        unsigned(align));
#endif
}



void
LLVM_Util::op_memset (llvm::Value *ptr, int val, llvm::Value *len, int align)
{
    builder().CreateMemSet (ptr, builder().getInt8((unsigned char)val), len,
#if OSL_LLVM_VERSION >= 100
        llvm::MaybeAlign(align));
#else
        unsigned(align));
#endif
}



void
LLVM_Util::op_memcpy (llvm::Value *dst, llvm::Value *src, int len, int align)
{
    op_memcpy (dst, align, src, align, len);
}



void
LLVM_Util::op_memcpy (llvm::Value *dst, int dstalign,
                      llvm::Value *src, int srcalign, int len)
{
#if OSL_LLVM_VERSION >= 100
    builder().CreateMemCpy (dst, llvm::MaybeAlign(dstalign), src, llvm::MaybeAlign(srcalign),
                            uint64_t(len));
#else
    builder().CreateMemCpy (dst, (unsigned)dstalign, src, (unsigned)srcalign,
                            uint64_t(len));
#endif
}



llvm::Value *
LLVM_Util::op_load (llvm::Value *ptr)
{
    return builder().CreateLoad (ptr);
}



llvm::Value *
LLVM_Util::op_linearize_16x_indices(llvm::Value *wide_index)
{
    llvm::Value* strided_indices = op_mul(wide_index, wide_constant(16, 16));
    llvm::Constant* offsets_to_lane[16] = {
        constant(0),
        constant(1),
        constant(2),
        constant(3),
        constant(4),
        constant(5),
        constant(6),
        constant(7),
        constant(8),
        constant(9),
        constant(10),
        constant(11),
        constant(12),
        constant(13),
        constant(14),
        constant(15),
    };
    llvm::Value *const_vec_offsets = llvm::ConstantVector::get(llvm::ArrayRef< llvm::Constant *>(&offsets_to_lane[0], 16));

    return op_add (strided_indices, const_vec_offsets);
}


llvm::Value *
LLVM_Util::op_linearize_8x_indices(llvm::Value *wide_index)
{
    llvm::Value* strided_indices = op_mul(wide_index, wide_constant(8, 8));
    llvm::Constant* offsets_to_lane[8] = {
        constant(0),
        constant(1),
        constant(2),
        constant(3),
        constant(4),
        constant(5),
        constant(6),
        constant(7)
    };
    llvm::Value *const_vec_offsets = llvm::ConstantVector::get(llvm::ArrayRef< llvm::Constant *>(&offsets_to_lane[0], 8));

    return op_add (strided_indices, const_vec_offsets);
}


std::array<llvm::Value *,2>
LLVM_Util::op_split_16x (llvm::Value * vector_val)
{
#if OSL_LLVM_VERSION >= 110
    using index_t = int32_t;
#else
    using index_t = uint32_t;
#endif
    const index_t extractLanes0_to_7[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const index_t extractLanes8_to_15[] = { 8, 9, 10, 11, 12, 13, 14, 15 };

    llvm::Value * half_vec_0 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes0_to_7));
    llvm::Value * half_vec_1 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes8_to_15));
    return {{half_vec_0, half_vec_1}};
}


std::array<llvm::Value *,2>
LLVM_Util::op_split_8x (llvm::Value * vector_val)
{
#if OSL_LLVM_VERSION >= 110
    using index_t = int32_t;
#else
    using index_t = uint32_t;
#endif
    const index_t extractLanes0_to_3[] = { 0, 1, 2, 3 };
    const index_t extractLanes4_to_7[] = { 4, 5, 6, 7 };

    llvm::Value * half_vec_0 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes0_to_3));
    llvm::Value * half_vec_1 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes4_to_7));
    return {{half_vec_0, half_vec_1}};
}


std::array<llvm::Value *,4>
LLVM_Util::op_quarter_16x (llvm::Value * vector_val)
{
    OSL_ASSERT(m_vector_width == 16);

#if OSL_LLVM_VERSION >= 110
    using index_t = int32_t;
#else
    using index_t = uint32_t;
#endif
    const index_t extractLanes0_to_3[] = { 0, 1, 2, 3 };
    const index_t extractLanes4_to_7[] = { 4, 5, 6, 7 };
    const index_t extractLanes8_to_11[] = { 8, 9, 10, 11};
    const index_t extractLanes12_to_15[] = { 12, 13, 14, 15 };

    llvm::Value * quarter_vec_0 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes0_to_3));
    llvm::Value * quarter_vec_1 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes4_to_7));
    llvm::Value * quarter_vec_2 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes8_to_11));
    llvm::Value * quarter_vec_3 = builder().CreateShuffleVector (vector_val, vector_val, llvm::makeArrayRef(extractLanes12_to_15));
    return {{quarter_vec_0, quarter_vec_1, quarter_vec_2, quarter_vec_3}};
}


llvm::Value *
LLVM_Util::op_combine_8x_vectors (llvm::Value * half_vec_1, llvm::Value * half_vec_2)
{
#if OSL_LLVM_VERSION >= 110
    using index_t = int32_t;
#else
    using index_t = uint32_t;
#endif
    static constexpr index_t combineIndices[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    return builder().CreateShuffleVector (half_vec_1, half_vec_2, llvm::makeArrayRef(combineIndices));
}

llvm::Value *
LLVM_Util::op_combine_4x_vectors (llvm::Value * half_vec_1, llvm::Value * half_vec_2)
{
#if OSL_LLVM_VERSION >= 110
    using index_t = int32_t;
#else
    using index_t = uint32_t;
#endif
    static constexpr index_t combineIndices[] = { 0, 1, 2, 3, 4, 5, 6, 7};
    return builder().CreateShuffleVector (half_vec_1, half_vec_2, llvm::makeArrayRef(combineIndices));
}



llvm::Value *
LLVM_Util::op_gather(llvm::Value *ptr, llvm::Value *wide_index)
{
    OSL_ASSERT(wide_index->getType() == type_wide_int());

    // To avoid loading masked off lanes, rather than add a bunch of
    // branches to skip loading of lanes, we assume accessing index 0 is
    // legal and in bounds and select the index and 0 based on the mask.
    // Because OSL owns the data layout of arrays, we can make this
    // assumption.  array[0] exists, is valid, and no indices will be
    // negative
    auto clamped_gather_from_uniform = [this, ptr, wide_index](llvm::Type *result_type)->llvm::Value * {
        llvm::Value *result = llvm::UndefValue::get(result_type);
        llvm::Value *clampedIndices = op_select(current_mask(), wide_index, wide_constant(0));
        for(int l=0; l < m_vector_width; ++l) {
            llvm::Value *index_for_lane = op_extract(clampedIndices, l);
            llvm::Value *address = GEP(ptr, index_for_lane);
            llvm::Value * val = op_load(address);
            result = op_insert (result, val, l);
        }
        return result;
    };

    auto clamped_gather_from_varying = [this, ptr, wide_index](llvm::Type *result_type)->llvm::Value * {
        llvm::Value *clampedIndices = op_select(current_mask(), wide_index, wide_constant(0));
        llvm::Value *result = llvm::UndefValue::get(result_type);
        for(int l=0; l < m_vector_width; ++l) {
            llvm::Value *index_for_lane = op_extract(clampedIndices, l);
            llvm::Value *wide_address = GEP(ptr, index_for_lane);
            llvm::Value *wide_val = op_load(wide_address);
            llvm::Value *val = op_extract(wide_val, l);
            result = op_insert (result, val, l);
        }
        return result;
    };

    if (ptr->getType() == type_int_ptr()) {
        if (m_supports_avx512f) {
            llvm::Function* func_avx512_gather_pi = nullptr;
            llvm::Value *int_mask = nullptr;
            switch(m_vector_width) {
                case 16:
                    int_mask = mask_as_int16(current_mask());
                    func_avx512_gather_pi = llvm::Intrinsic::getDeclaration (module(),
                                        llvm::Intrinsic::x86_avx512_gather_dpi_512);
                    break;
                case 8:
                    int_mask = mask_as_int8(current_mask());
                    func_avx512_gather_pi = llvm::Intrinsic::getDeclaration (module(),
                            llvm::Intrinsic::x86_avx512_gather3siv8_si);
                    break;
                default:
                    OSL_ASSERT(0 && "unsupported native bit mask width");
            };
            OSL_ASSERT(int_mask);
            OSL_ASSERT(func_avx512_gather_pi);

            llvm::Value *unmasked_value = wide_constant(0);
            llvm::Value *args[] = {
                unmasked_value,
                void_ptr(ptr),
                wide_index,
                int_mask,
                constant(4)
            };
            return builder().CreateCall (func_avx512_gather_pi, makeArrayRef(args));
        } else if (m_supports_avx2) {
            llvm::Function* func_avx2_gather_pi = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_avx2_gather_d_d_256);
            OSL_ASSERT(func_avx2_gather_pi);

            llvm::Constant *avx2_unmasked_value = wide_constant(8, 0);

            // Convert <8 x i1> -> <8 x i32>
            llvm::Value * wide_int_mask = builder().CreateSExt(current_mask(), type_wide_int());
            switch(m_vector_width) {
                case 16:
                {
                    // Convert <16 x i32> -> to <2 x< 8 x i32>>
                    auto w8_int_masks = op_split_16x(wide_int_mask);
                    auto w8_int_indices = op_split_16x(wide_index);
                    llvm::Value *args[] = {
                        avx2_unmasked_value,
                        void_ptr(ptr),
                        w8_int_indices[0],
                        w8_int_masks[0],
                        constant8(4)
                    };
                    llvm::Value *gather1 = builder().CreateCall (func_avx2_gather_pi, makeArrayRef(args));
                    args[2] = w8_int_indices[1];
                    args[3] = w8_int_masks[1];
                    llvm::Value *gather2 = builder().CreateCall (func_avx2_gather_pi, makeArrayRef(args));
                    return op_combine_8x_vectors(gather1,gather2);
                }
                case 8:
                {
                    llvm::Value *args[] = {
                        avx2_unmasked_value,
                        void_ptr(ptr),
                        wide_index,
                        wide_int_mask,
                        constant8(4)
                    };
                    llvm::Value *gather_result = builder().CreateCall (func_avx2_gather_pi, makeArrayRef(args));
                    return gather_result;
                }
                default:
                    OSL_ASSERT(0 && "unsupported width");
            };
        } else {
            return clamped_gather_from_uniform(type_wide_int());
        }

    } else if (ptr->getType() == type_float_ptr()) {
        if (m_supports_avx512f) {
            llvm::Function* func_avx512_gather_ps = nullptr;
            llvm::Value *int_mask = nullptr;
            switch(m_vector_width) {
                case 16:
                    int_mask = mask_as_int16(current_mask());
                    func_avx512_gather_ps = llvm::Intrinsic::getDeclaration (module(),
                            llvm::Intrinsic::x86_avx512_gather_dps_512);
                    break;
                case 8:
                    int_mask = mask_as_int8(current_mask());
                    func_avx512_gather_ps = llvm::Intrinsic::getDeclaration (module(),
                            llvm::Intrinsic::x86_avx512_gather3siv8_sf);
                    break;
                default:
                    OSL_ASSERT(0 && "unsupported native bit mask width");
            };
            OSL_ASSERT(int_mask);
            OSL_ASSERT(func_avx512_gather_ps);

            llvm::Value *unmasked_value = wide_constant(0.0f);
            llvm::Value *args[] = {
                unmasked_value,
                void_ptr(ptr),
                wide_index,
                int_mask,
                constant(4) // not sure why the scale
            };
            return builder().CreateCall (func_avx512_gather_ps, makeArrayRef(args));
        } else if (m_supports_avx2) {
            llvm::Function* func_avx2_gather_ps = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_avx2_gather_d_ps_256);
            OSL_ASSERT(func_avx2_gather_ps);

            llvm::Constant* avx2_unmasked_value = wide_constant(8, 0.0f);

            // Convert <? x i1> -> <?x i32> -> to
            llvm::Value * wide_int_mask = builder().CreateSExt(current_mask(), type_wide_int());
            switch(m_vector_width) {
                case 16:
                {
                    // Convert <16 x i32> -> to <2 x< 8 x i32>>
                    auto w8_int_masks = op_split_16x(wide_int_mask);
                    auto w8_int_indices = op_split_16x(wide_index);
                    llvm::Value *args[] = {
                            avx2_unmasked_value,
                        void_ptr(ptr),
                        w8_int_indices[0],
                        builder().CreateBitCast (w8_int_masks[0], llvm_vector_type(type_float(), 8)),
                        constant8(4)
                    };
                    llvm::Value *gather1 = builder().CreateCall (func_avx2_gather_ps, makeArrayRef(args));
                    args[2] = w8_int_indices[1];
                    args[3] = builder().CreateBitCast (w8_int_masks[1], llvm_vector_type(type_float(), 8));
                    llvm::Value *gather2 = builder().CreateCall (func_avx2_gather_ps, makeArrayRef(args));
                    return op_combine_8x_vectors(gather1,gather2);
                }
                case 8:
                {
                    llvm::Value *args[] = {
                            avx2_unmasked_value,
                        void_ptr(ptr),
                        wide_index,
                        builder().CreateBitCast (wide_int_mask, llvm_vector_type(type_float(), 8)),
                        constant8(4)
                    };
                    llvm::Value *gather = builder().CreateCall (func_avx2_gather_ps, makeArrayRef(args));
                    return gather;
                }
            }
        } else {
            return clamped_gather_from_uniform(type_wide_float());
        }
    } else if (ptr->getType() == type_ustring_ptr()) {
        if (m_supports_avx512f) {
            // TODO:  Are we guaranteed a 64bit pointer?
            switch(m_vector_width) {
                case 16:
                {
                    // Gather 64bit integer, as that is binary compatible with 64bit pointers of ustring
                    llvm::Function* func_avx512_gather_dpq = llvm::Intrinsic::getDeclaration (module(),
                            llvm::Intrinsic::x86_avx512_gather_dpq_512);
                    OSL_ASSERT(func_avx512_gather_dpq);

                    // We can only gather 8 at a time, so need to split the work over 2 gathers
                    auto w8_bit_masks = op_split_16x(current_mask());
                    auto w8_int_indices = op_split_16x(wide_index);

                    llvm::Value *unmasked_value = builder().CreateVectorSplat(8,constant64(0));
                    llvm::Value *args[] = {
                        unmasked_value,
                        void_ptr(ptr),
                        w8_int_indices[0],
                        mask_as_int8(w8_bit_masks[0]),
                        constant(8)
                    };
                    llvm::Value *gather1 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));
                    args[2] = w8_int_indices[1];
                    args[3] = mask_as_int8(w8_bit_masks[1]);
                    llvm::Value *gather2 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));

                    return builder().CreateIntToPtr(op_combine_8x_vectors(gather1, gather2), type_wide_string());
                }
                case 8:
                {
                    // Gather 64bit integer, as that is binary compatible with 64bit pointers of ustring
                    llvm::Function* func_avx512_gather_dpq = llvm::Intrinsic::getDeclaration (module(),
                            llvm::Intrinsic::x86_avx512_gather3siv4_di);
                    OSL_ASSERT(func_avx512_gather_dpq);

                    // We can only gather 4 at a time, so need to split the work over 2 gathers
                    auto w4_bit_masks = op_split_8x(current_mask());
                    auto w4_int_indices = op_split_8x(wide_index);

                    llvm::Value *unmasked_value = builder().CreateVectorSplat(4,constant64(0));
                    llvm::Value *args[] = {
                        unmasked_value,
                        void_ptr(ptr),
                        w4_int_indices[0],
                        mask4_as_int8(w4_bit_masks[0]),
                        constant(8)
                    };
                    llvm::Value *gather1 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));
                    args[2] = w4_int_indices[1];
                    args[3] = mask4_as_int8(w4_bit_masks[1]);
                    llvm::Value *gather2 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));

                    return builder().CreateIntToPtr(op_combine_4x_vectors(gather1, gather2), type_wide_string());
                }
                default:
                    OSL_ASSERT(0 && "unsupported native bit mask width");
            }
        } else {
            return clamped_gather_from_uniform(type_wide_string());
        }
    } else if (ptr->getType() == type_wide_float_ptr()) {
        if (m_supports_avx512f) {
            switch(m_vector_width) {
            case 16:
            {
                llvm::Function* func_avx512_gather_ps = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_gather_dps_512);
                OSL_ASSERT(func_avx512_gather_ps);

                llvm::Value *unmasked_value = wide_constant(0.0f);
                llvm::Value *args[] = {
                    unmasked_value,
                    void_ptr(ptr),
                    op_linearize_16x_indices(wide_index),
                    mask_as_int16(current_mask()),
                    constant(4)
                };
                return builder().CreateCall (func_avx512_gather_ps, makeArrayRef(args));
            }
            case 8:
            {
                llvm::Function* func_avx512_gather_ps = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_gather3siv8_sf);
                OSL_ASSERT(func_avx512_gather_ps);

                llvm::Value *unmasked_value = wide_constant(0.0f);
                llvm::Value *args[] = {
                    unmasked_value,
                    void_ptr(ptr),
                    op_linearize_8x_indices(wide_index),
                    mask_as_int8(current_mask()),
                    constant(4)
                };
                return builder().CreateCall (func_avx512_gather_ps, makeArrayRef(args));
            }
            default:
                OSL_ASSERT(0 && "unsupported native bit mask width");
            };

        } else if (m_supports_avx2) {
            llvm::Function* func_avx2_gather_ps = llvm::Intrinsic::getDeclaration (module(),
                    llvm::Intrinsic::x86_avx2_gather_d_ps_256);
            OSL_ASSERT(func_avx2_gather_ps);

            llvm::Constant *avx2_unmasked_value = wide_constant(8, 0.0f);
            // Convert <? x i1> -> <? x i32>
            llvm::Value * wide_int_mask = builder().CreateSExt(current_mask(), type_wide_int());
            switch(m_vector_width) {
            case 16:
            {
                // Convert <16 x i32> -> to <2 x< 8 x i32>>
                auto w8_int_masks = op_split_16x(wide_int_mask);
                auto w8_int_indices = op_split_16x(op_linearize_16x_indices(wide_index));
                llvm::Value *args[] = {
                    avx2_unmasked_value,
                    void_ptr(ptr),
                    w8_int_indices[0],
                    builder().CreateBitCast (w8_int_masks[0], llvm_vector_type(type_float(), 8)),
                    constant8(4)
                };
                llvm::Value *gather1 = builder().CreateCall (func_avx2_gather_ps, makeArrayRef(args));
                args[2] = w8_int_indices[1];
                args[3] = builder().CreateBitCast (w8_int_masks[1], llvm_vector_type(type_float(), 8));
                llvm::Value *gather2 = builder().CreateCall (func_avx2_gather_ps, makeArrayRef(args));
                return op_combine_8x_vectors(gather1, gather2);
            }
            case 8:
            {
                auto int_indices = op_linearize_8x_indices(wide_index);
                llvm::Value *args[] = {
                    avx2_unmasked_value,
                    void_ptr(ptr),
                    int_indices,
                    builder().CreateBitCast (wide_int_mask, llvm_vector_type(type_float(), 8)),
                    constant8(4)
                };
                llvm::Value *gather_result = builder().CreateCall (func_avx2_gather_ps, makeArrayRef(args));
                return gather_result;
            }
            default:
                OSL_ASSERT(0 && "unsupported vector width for avx2 gather");
            }
        } else {
            return clamped_gather_from_varying(type_wide_float());
        }
    } else if (ptr->getType() == type_wide_int_ptr()) {
        if (m_supports_avx512f) {
            switch(m_vector_width) {
            case 16:
            {
                llvm::Function* func_avx512_gather_pi = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_gather_dpi_512);
                OSL_ASSERT(func_avx512_gather_pi);

                llvm::Value *unmasked_value = wide_constant(0);
                llvm::Value *args[] = {
                    unmasked_value,
                    void_ptr(ptr),
                    op_linearize_16x_indices(wide_index),
                    mask_as_int16(current_mask()),
                    constant(4)
                };
                return builder().CreateCall (func_avx512_gather_pi, makeArrayRef(args));
            }
            case 8:
            {
                llvm::Function* func_avx512_gather_pi = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_gather3siv8_si);
                OSL_ASSERT(func_avx512_gather_pi);

                llvm::Value *unmasked_value = wide_constant(0);
                llvm::Value *args[] = {
                    unmasked_value,
                    void_ptr(ptr),
                    op_linearize_8x_indices(wide_index),
                    mask_as_int8(current_mask()),
                    constant(4)
                };
                return builder().CreateCall (func_avx512_gather_pi, makeArrayRef(args));
            }
            default:
                OSL_ASSERT(0 && "unsupported native bit mask width");
            }
        } else if (m_supports_avx2) {
            switch(m_vector_width) {
            case 16:
            {
                llvm::Function* func_avx2_gather_pi = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx2_gather_d_d_256);
                OSL_ASSERT(func_avx2_gather_pi);

                llvm::Constant *avx2_unmasked_value = wide_constant(8, 0);

                // Convert <16 x i1> -> <16 x i32> -> to <2 x< 8 x i32>>
                llvm::Value * wide_int_mask = builder().CreateSExt(current_mask(), type_wide_int());
                auto w8_int_masks = op_split_16x(wide_int_mask);
                auto w8_int_indices = op_split_16x(op_linearize_16x_indices(wide_index));
                llvm::Value *args[] = {
                    avx2_unmasked_value,
                    void_ptr(ptr),
                    w8_int_indices[0],
                    w8_int_masks[0],
                    constant8(4)
                };
                llvm::Value *gather1 = builder().CreateCall (func_avx2_gather_pi, makeArrayRef(args));
                args[2] = w8_int_indices[1];
                args[3] = w8_int_masks[1];
                llvm::Value *gather2 = builder().CreateCall (func_avx2_gather_pi, makeArrayRef(args));
                return op_combine_8x_vectors(gather1, gather2);
            }
            case 8:
            {
                llvm::Function* func_avx2_gather_pi = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx2_gather_d_d_256);
                OSL_ASSERT(func_avx2_gather_pi);

                llvm::Constant *avx2_unmasked_value = wide_constant(8, 0);

                // Convert <16 x i1> -> <16 x i32> -> to <2 x< 8 x i32>>
                llvm::Value * wide_int_mask = builder().CreateSExt(current_mask(), type_wide_int());
                auto int_indices = op_linearize_8x_indices(wide_index);
                llvm::Value *args[] = {
                    avx2_unmasked_value,
                    void_ptr(ptr),
                    int_indices,
                    wide_int_mask,
                    constant8(4)
                };
                llvm::Value *gather_result = builder().CreateCall (func_avx2_gather_pi, makeArrayRef(args));
                return gather_result;
            }
            default:
                OSL_ASSERT(0 && "unsupported vector width for avx2 gather");
            }
        } else {
            return clamped_gather_from_varying(type_wide_int());
        }
    } else if (ptr->getType() == llvm::PointerType::get(type_wide_string(),0)) {
        if (m_supports_avx512f) {
            // TODO:  Are we guaranteed a 64bit pointer?
            switch(m_vector_width) {
            case 16:
            {
                // Gather 64bit integer, as that is binary compatible with
                // 64bit pointers of ustring
                llvm::Function* func_avx512_gather_dpq = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_gather_dpq_512);
                OSL_ASSERT(func_avx512_gather_dpq);

                // We can only gather 8 at a time, so need to split the work
                // over 2 gathers
                auto w8_bit_masks = op_split_16x(current_mask());
                auto w8_int_indices = op_split_16x(op_linearize_16x_indices(wide_index));

                llvm::Value *unmasked_value = builder().CreateVectorSplat(8,constant64(0));
                llvm::Value *args[] = {
                    unmasked_value,
                    void_ptr(ptr),
                    w8_int_indices[0],
                    mask_as_int8(w8_bit_masks[0]),
                    constant(8)
                };
                llvm::Value *gather1 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));
                args[2] = w8_int_indices[1];
                args[3] = mask_as_int8(w8_bit_masks[1]);
                llvm::Value *gather2 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));

                return builder().CreateIntToPtr(op_combine_8x_vectors(gather1, gather2), type_wide_string());
            }
            case 8:
            {
                // Gather 64bit integer, as that is binary compatible with 64bit pointers of ustring
                llvm::Function* func_avx512_gather_dpq = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_gather3siv4_di);
                OSL_ASSERT(func_avx512_gather_dpq);

                // TODO: we technically could gather all 8 if we let a
                // 512bit operation run, but that could drop the clock
                // frequency.  If a renderer were both executing 16wide
                // AVX512 and 8wide AVX512, then using the 512 bit
                // instruction wouldn't hurt as the 16wide code would
                // already have incurred the clock drop. In future, maybe
                // add an option to enable 512 bit operations when vector
                // width is 8.

                // We can only gather 4 at a time, so need to split the work
                // over 2 gathers
                auto w4_bit_masks = op_split_8x(current_mask());
                auto w4_int_indices = op_split_8x(op_linearize_8x_indices(wide_index));

                llvm::Value *unmasked_value = builder().CreateVectorSplat(4,constant64(0));
                llvm::Value *args[] = {
                    unmasked_value,
                    void_ptr(ptr),
                    w4_int_indices[0],
                    mask4_as_int8(w4_bit_masks[0]),
                    constant(8)
                };
                llvm::Value *gather1 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));
                args[2] = w4_int_indices[1];
                args[3] = mask4_as_int8(w4_bit_masks[1]);
                llvm::Value *gather2 = builder().CreateCall (func_avx512_gather_dpq, makeArrayRef(args));

                return builder().CreateIntToPtr(op_combine_4x_vectors(gather1, gather2), type_wide_string());
            }
            default:
                OSL_ASSERT(0 && "unsupported native bit mask width");
            }

        } else {
            // AVX2 case falls through to here, choose not to specialize and use
            // generic code gen as 4 AVX2 gathers would be required
            return clamped_gather_from_varying(type_wide_string());
        }

    } else {
        std::cout << "ptr->getType() = " <<
        llvm_typenameof(ptr) <<
        std::endl;

        OSL_ASSERT(0 && "unsupported ptr type");
    }
    return nullptr;
}



void
LLVM_Util::op_scatter(llvm::Value *wide_val, llvm::Value *ptr, llvm::Value *wide_index)
{
    OSL_ASSERT(wide_index->getType() == type_wide_int());

    auto scatter_using_conditional_block_per_lane = [this, wide_val, wide_index](llvm::Value * cast_ptr)->void {
        llvm::Value * linear_indices = nullptr;
        switch(m_vector_width) {
        case 16:
            linear_indices = op_linearize_16x_indices(wide_index);
            break;
        case 8:
            linear_indices = op_linearize_8x_indices(wide_index);
            break;
        default:
            OSL_ASSERT(0 && "unsupported vector width for scatter");
        };

        llvm::BasicBlock* test_scatter_per_lane[MaxSupportedSimdLaneCount+1];
        for(int l=0; l < m_vector_width; ++l) {
            test_scatter_per_lane[l] = new_basic_block (std::string("test scatter lane=").append(std::to_string(l)));
        }
        test_scatter_per_lane[m_vector_width] = new_basic_block ("after scatter");

        // Main performance strategy is to not perform any extractions inside the conditional section
        llvm::Value *val_per_lane[MaxSupportedSimdLaneCount];
        for(int l=0; l < m_vector_width; ++l) {
            val_per_lane[l] = op_extract(wide_val, l);
        }
        llvm::Value *cm = current_mask();
        llvm::Value *mask_per_lane[MaxSupportedSimdLaneCount];
        for(int l=0; l < m_vector_width; ++l) {
            mask_per_lane[l] = op_extract(cm, l);
        }

        llvm::Value *index_per_lane[MaxSupportedSimdLaneCount];
        for(int l=0; l < m_vector_width; ++l) {
            index_per_lane[l] = op_extract(linear_indices, l);
        }

        op_branch(test_scatter_per_lane[0]);
        for(int l=0; l < m_vector_width; ++l) {
            llvm::BasicBlock* scatter_block = new_basic_block (std::string("scatter lane=").append(std::to_string(l)));
            op_branch(mask_per_lane[l], scatter_block, test_scatter_per_lane[l+1]);

            llvm::Value *address = GEP(cast_ptr, index_per_lane[l]);
            // uniform store, no need to mess with masking
            op_store(val_per_lane[l], address);
            op_branch(test_scatter_per_lane[l+1]);
        }
    };

    if (ptr->getType() == type_wide_float_ptr()) {
        OSL_ASSERT(wide_val->getType() == type_wide_float());
        if (m_supports_avx512f) {
            switch(m_vector_width) {
            case 16:
            {
                llvm::Function* func_avx512_scatter_ps = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_scatter_dps_512);
                OSL_ASSERT(func_avx512_scatter_ps);

                llvm::Value *args[] = {
                    void_ptr(ptr),
                    mask_as_int16(current_mask()),
                    op_linearize_16x_indices(wide_index),
                    wide_val,
                    constant(4)
                };
                builder().CreateCall (func_avx512_scatter_ps, makeArrayRef(args));
                return;
            }
            case 8:
            {
                llvm::Function* func_avx512_scatter_ps = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_scattersiv8_sf);
                OSL_ASSERT(func_avx512_scatter_ps);

                llvm::Value *args[] = {
                    void_ptr(ptr),
                    mask_as_int8(current_mask()),
                    op_linearize_8x_indices(wide_index),
                    wide_val,
                    constant(4)
                };
                builder().CreateCall (func_avx512_scatter_ps, makeArrayRef(args));
                return;
            }
            default:
                OSL_ASSERT(0 && "incomplete vector width for AVX512 scatter");
            }
        } else {
            // AVX2, AVX, SSE4.2 fall through to here
            llvm::Value * float_ptr =  builder().CreatePointerBitCastOrAddrSpaceCast(ptr, type_float_ptr());
            scatter_using_conditional_block_per_lane(float_ptr);
            return;
        }
    }
    if (ptr->getType() == type_wide_int_ptr()) {
        OSL_ASSERT(wide_val->getType() == type_wide_int());
        if (m_supports_avx512f) {
            switch(m_vector_width) {
            case 16:
            {
                llvm::Function* func_avx512_scatter_pi = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_scatter_dpi_512);
                OSL_ASSERT(func_avx512_scatter_pi);

                llvm::Value *args[] = {
                    void_ptr(ptr),
                    mask_as_int16(current_mask()),
                    op_linearize_16x_indices(wide_index),
                    wide_val,
                    constant(4)
                };
                builder().CreateCall (func_avx512_scatter_pi, makeArrayRef(args));
                return;
            }
            case 8:
            {
                llvm::Function* func_avx512_scatter_pi = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_scattersiv8_si);
                OSL_ASSERT(func_avx512_scatter_pi);

                llvm::Value *args[] = {
                    void_ptr(ptr),
                    mask_as_int8(current_mask()),
                    op_linearize_8x_indices(wide_index),
                    wide_val,
                    constant(4)
                };
                builder().CreateCall (func_avx512_scatter_pi, makeArrayRef(args));
                return;
            }
            default:
                OSL_ASSERT(0 && "incomplete vector width for AVX512 scatter");
            }
        } else {
            // AVX2, AVX, SSE4.2 fall through to here
            llvm::Value * int_ptr =  builder().CreatePointerBitCastOrAddrSpaceCast(ptr, type_int_ptr());
            scatter_using_conditional_block_per_lane(int_ptr);
            return;
        }
    } else if (ptr->getType() == llvm::PointerType::get(type_wide_string(),0)) {
        OSL_ASSERT(wide_val->getType() == type_wide_string());
        if (m_supports_avx512f) {

            switch(m_vector_width) {
            case 16:
            {
                llvm::Value * linear_indices = op_linearize_16x_indices(wide_index);

                llvm::Function* func_avx512_scatter_dpq = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_scatter_dpq_512);
                OSL_ASSERT(func_avx512_scatter_dpq);

                // We can only scatter 8 at a time, so need to split the
                // work over 2 scatters
                llvm::Type * w8_address_int = llvm_vector_type(type_addrint(), 8);

                auto w8_bit_masks = op_split_16x(current_mask());
                auto w8_int_indices = op_split_16x(linear_indices);
                auto w8_string_vals = op_split_16x(wide_val);
                std::array<llvm::Value *,2> w8_address_int_val = {{
                        builder().CreatePtrToInt(w8_string_vals[0], w8_address_int),
                        builder().CreatePtrToInt(w8_string_vals[1], w8_address_int)
                }};

                llvm::Value *args[] = {
                    void_ptr(ptr),
                    mask_as_int8(w8_bit_masks[0]),
                    w8_int_indices[0],
                    w8_address_int_val[0],
                    constant(8)
                };
                builder().CreateCall (func_avx512_scatter_dpq, makeArrayRef(args));
                args[1] = mask_as_int8(w8_bit_masks[1]);
                args[2] = w8_int_indices[1];
                args[3] = w8_address_int_val[1];
                builder().CreateCall (func_avx512_scatter_dpq, makeArrayRef(args));
                return;
            }
            case 8:
            {
                llvm::Value * linear_indices = op_linearize_8x_indices(wide_index);

                llvm::Function* func_avx512_scatter_dpq = llvm::Intrinsic::getDeclaration (module(),
                        llvm::Intrinsic::x86_avx512_scatter_dpq_512);
                OSL_ASSERT(func_avx512_scatter_dpq);

                llvm::Type * wide_address_int_type = llvm_vector_type(type_addrint(), 8);
                llvm::Value * address_int_val = builder().CreatePtrToInt(wide_val, wide_address_int_type);

                llvm::Value *args[] = {
                    void_ptr(ptr),
                    mask_as_int8(current_mask()),
                    linear_indices,
                    address_int_val,
                    constant(8)
                };
                builder().CreateCall (func_avx512_scatter_dpq, makeArrayRef(args));
                return;
            }
            default:
                OSL_ASSERT(0 && "incomplete vector width for AVX512 scatter");
            }
        } else {
            // AVX2, AVX, SSE4.2 fall through to here
            llvm::Value * ustring_ptr =  builder().CreatePointerBitCastOrAddrSpaceCast(ptr, type_ustring_ptr());
            scatter_using_conditional_block_per_lane(ustring_ptr);
            return;
        }
    }

    std::cout << "ptr->getType() = " <<
    llvm_typenameof(ptr) <<
    std::endl;

    OSL_ASSERT(0 && "unsupported ptr type");
}



void
LLVM_Util::push_mask(llvm::Value *mask, bool negate, bool absolute)
{
    OSL_ASSERT(mask->getType() == type_wide_bool());
    if (m_mask_stack.empty()) {
        m_mask_stack.push_back(MaskInfo{mask, negate, 0});
    } else {

        MaskInfo & mi = m_mask_stack.back();
        llvm::Value *prev_mask = mi.mask;
        bool prev_negate = mi.negate;

        int applied_return_mask_count = absolute? 0 : mi.applied_return_mask_count;

        if (false == prev_negate) {
            if (false == negate)
            {
                llvm::Value *blended_mask;
                if (absolute) {
                    blended_mask = mask;
                } else {
                    blended_mask = builder().CreateSelect(prev_mask, mask, prev_mask);
                }
                m_mask_stack.push_back(MaskInfo{blended_mask, false, applied_return_mask_count});
            } else {
                OSL_ASSERT(false == absolute);
                llvm::Value *blended_mask = builder().CreateSelect(mask, wide_constant_bool(false), prev_mask);
                m_mask_stack.push_back(MaskInfo{blended_mask, false, applied_return_mask_count});
            }
        } else {
            if (false == negate)
            {
                llvm::Value *blended_mask;
                if (absolute) {
                    blended_mask = mask;
                } else {
                    blended_mask = builder().CreateSelect(prev_mask, wide_constant_bool(false), mask);
                }
                m_mask_stack.push_back(MaskInfo{blended_mask, false, applied_return_mask_count});
            } else {
                OSL_ASSERT(false == absolute);
                llvm::Value *blended_mask = builder().CreateSelect(prev_mask, prev_mask, mask);
                m_mask_stack.push_back(MaskInfo{blended_mask, true, applied_return_mask_count});
            }
        }
    }
}



llvm::Value *
LLVM_Util::shader_mask()
{
    llvm::Value * loc_of_shader_mask = masked_shader_context().location_of_mask;
    return op_load_mask(loc_of_shader_mask);
}



void
LLVM_Util::apply_exit_to_mask_stack()
{
    OSL_ASSERT (false == m_mask_stack.empty());


    llvm::Value * loc_of_shader_mask = masked_shader_context().location_of_mask;
    llvm::Value * shader_mask = op_load_mask(loc_of_shader_mask);

    llvm::Value * loc_of_function_mask = masked_function_context().location_of_mask;
    llvm::Value * function_mask = op_load_mask(loc_of_function_mask);

    // For any inactive lanes of the shader mask set the function_mask to
    // 0.
    llvm::Value * modified_function_mask = builder().CreateSelect(shader_mask, function_mask, shader_mask);

    op_store_mask(modified_function_mask, loc_of_function_mask);

    // Apply the modified_function_mask to the current conditional mask
    // stack By bumping the return count, the modified_return_mask will get
    // applied to the conditional mask stack as it unwinds.
    masked_function_context().return_count++;

    // We could just call apply_return_to_mask_stack(), but will repeat the
    // work here to take advantage of the already loaded return mask.
    auto & mi = m_mask_stack.back();

    int masked_return_count = masked_function_context().return_count;
    OSL_ASSERT(masked_return_count > mi.applied_return_mask_count);
    llvm::Value * existing_mask = mi.mask;

    if(mi.negate) {
        mi.mask = builder().CreateSelect(modified_function_mask, existing_mask, wide_constant_bool(true));
    } else {
        mi.mask = builder().CreateSelect(modified_function_mask, existing_mask, modified_function_mask);
    }
    mi.applied_return_mask_count = masked_return_count;
}



void
LLVM_Util::apply_return_to_mask_stack()
{
    OSL_ASSERT (false == m_mask_stack.empty());

    auto & mi = m_mask_stack.back();
    int masked_return_count = masked_function_context().return_count;
    // TODO: might be impossible for this conditional to be false, could
    // change to assert or remove applied_return_mask_count entirely.
    if (masked_return_count > mi.applied_return_mask_count) {
        llvm::Value * existing_mask = mi.mask;

        llvm::Value * loc_of_return_mask = masked_function_context().location_of_mask;
        llvm::Value * rs_mask = op_load_mask(loc_of_return_mask);
        if(mi.negate) {
            mi.mask = builder().CreateSelect(rs_mask, existing_mask, wide_constant_bool(true));
        } else {
            mi.mask = builder().CreateSelect(rs_mask, existing_mask, rs_mask);
        }
        mi.applied_return_mask_count = masked_return_count;
    }

}



void
LLVM_Util::apply_break_to_mask_stack()
{
    OSL_ASSERT (false == m_mask_stack.empty());

    auto & mi = m_mask_stack.back();

    llvm::Value * existing_mask = mi.mask;

    // The loop control flow should already have the break applied, So we
    // can load it, then use its value to restrict the current stack mask.
    llvm::Value * loc_of_cond_mask = masked_loop_context().location_of_control_mask;
    llvm::Value * cond_mask = op_load_mask(loc_of_cond_mask);
    if(mi.negate) {
        mi.mask = builder().CreateSelect(cond_mask, existing_mask, wide_constant_bool(true));
    } else {
        mi.mask = builder().CreateSelect(cond_mask, existing_mask, cond_mask);
    }
}



void
LLVM_Util::apply_continue_to_mask_stack()
{
    OSL_ASSERT (false == m_mask_stack.empty());

    auto & mi = m_mask_stack.back();

    llvm::Value * existing_mask = mi.mask;

    // The continue mask is tracking which lanes have had continue within
    // this loop.  So we can load it, then use its value to restrict the
    // current stack mask.
    // NOTE: a true value for a lane means that continue was called
    llvm::Value * loc_of_continue_mask = masked_loop_context().location_of_continue_mask;
    llvm::Value * continue_mask = op_load_mask(loc_of_continue_mask);
    if(mi.negate) {
        mi.mask = builder().CreateSelect(continue_mask, wide_constant_bool(true), existing_mask);
    } else {
        mi.mask = builder().CreateSelect(continue_mask, wide_constant_bool(false), existing_mask);
    }
}



llvm::Value *
LLVM_Util::apply_return_to(llvm::Value *existing_mask)
{
    // caller should have checked masked_return_count() beforehand
    OSL_ASSERT (masked_function_context().return_count > 0);

    llvm::Value * loc_of_return_mask = masked_function_context().location_of_mask;
    llvm::Value * rs_mask = op_load_mask(loc_of_return_mask);
    llvm::Value *result = builder().CreateSelect(rs_mask, existing_mask, rs_mask);
    return result;
}



void
LLVM_Util::pop_mask()
{
    OSL_ASSERT(false == m_mask_stack.empty());

    m_mask_stack.pop_back();
}



llvm::Value *
LLVM_Util::current_mask()
{
    OSL_ASSERT(!m_mask_stack.empty());
    auto & mi = m_mask_stack.back();
    if (mi.negate) {
        llvm::Value *negated_mask = builder().CreateSelect(mi.mask, wide_constant_bool(false), wide_constant_bool(true));
        return negated_mask;
    } else {
        return mi.mask;
    }
}



void
LLVM_Util::op_masked_break()
{
    OSL_DEV_ONLY(std::cout << "op_masked_break" << std::endl);

    OSL_ASSERT(false == m_mask_stack.empty());

    const MaskInfo & mi = m_mask_stack.back();
    // Because we are inside a conditional branch we can't let our local
    // modified mask be directly used by other scopes, instead we must store
    // the result of to the stack for the outer scope to pick up and use.
    auto & loop = masked_loop_context();
    llvm::Value * loc_of_cond_mask = loop.location_of_control_mask;

    llvm::Value * cond_mask = op_load_mask(loc_of_cond_mask);

    llvm::Value * break_from_mask = mi.mask;
    llvm::Value * new_cond_mask;

    // For any active lanes of the mask we are returning from
    // set the after_if_mask to 0.
    if (mi.negate) {
        new_cond_mask = builder().CreateSelect(break_from_mask, cond_mask, break_from_mask);
    } else {
        new_cond_mask = builder().CreateSelect(break_from_mask, wide_constant_bool(false), cond_mask);
    }

    op_store_mask(new_cond_mask, loc_of_cond_mask);

    // Track that a break was called in the current masked loop
    loop.break_count++;
}



void
LLVM_Util::op_masked_continue()
{
    OSL_DEV_ONLY(std::cout << "op_masked_break" << std::endl);

    OSL_ASSERT(false == m_mask_stack.empty());

    const MaskInfo & mi = m_mask_stack.back();
    // Because we are inside a conditional branch
    // we can't let our local modified mask be directly used
    // by other scopes, instead we must store the result
    // of to the stack for the outer scope to pickup and
    // use
    auto & loop = masked_loop_context();
    llvm::Value * loc_of_continue_mask = loop.location_of_continue_mask;

    llvm::Value * continue_mask = op_load_mask(loc_of_continue_mask);

    llvm::Value * continue_from_mask = mi.mask;
    llvm::Value * new_abs_continue_mask;

    // For any active lanes of the mask we are returning from
    // set the after_if_mask to 0.
    if (mi.negate) {
        new_abs_continue_mask = builder().CreateSelect(continue_from_mask, continue_mask, this->wide_constant_bool(true));
    } else {
        new_abs_continue_mask = builder().CreateSelect(continue_from_mask, continue_from_mask, continue_mask);
    }

    op_store_mask(new_abs_continue_mask, loc_of_continue_mask);

    // Track that a break was called in the current masked loop
    loop.continue_count++;
}



void
LLVM_Util::op_masked_exit()
{
    OSL_DEV_ONLY(std::cout << "push_mask_exit" << std::endl);

    OSL_ASSERT(false == m_mask_stack.empty());

    const MaskInfo & mi = m_mask_stack.back();
    llvm::Value * exit_from_mask = mi.mask;

    // Because we are inside a conditional branch we can't let our local
    // modified mask be directly used by other scopes, instead we must store
    // the result of to the stack for the outer scope to pick up and use.
    {
        llvm::Value * loc_of_shader_mask = masked_shader_context().location_of_mask;
        llvm::Value * shader_mask = op_load_mask(loc_of_shader_mask);

        llvm::Value * modifiedMask;
        // For any active lanes of the mask we are returning from set the
        // shader scope mask to 0.
        if (mi.negate) {
            modifiedMask = builder().CreateSelect(exit_from_mask, shader_mask, exit_from_mask);
        } else {
            modifiedMask = builder().CreateSelect(exit_from_mask, wide_constant_bool(false), shader_mask);
        }

        op_store_mask(modifiedMask, loc_of_shader_mask);
    }

    // Are we inside a function scope, then we will need to modify its
    // active lane mask functions higher up in the stack will apply the
    // current exit mask when functions are popped.
    if (inside_of_inlined_masked_function_call()) {
        llvm::Value * loc_of_function_mask = masked_function_context().location_of_mask;
        llvm::Value * function_mask = op_load_mask(loc_of_function_mask);


        llvm::Value * modifiedMask;

        // For any active lanes of the mask we are returning from
        // set the after_if_mask to 0.
        if (mi.negate) {
            modifiedMask = builder().CreateSelect(exit_from_mask, function_mask, exit_from_mask);
        } else {
            modifiedMask = builder().CreateSelect(exit_from_mask, wide_constant_bool(false), function_mask);
        }

        op_store_mask(modifiedMask, loc_of_function_mask);
    }

    // Bumping the masked exit count will cause the exit mask to be applied
    // to the return mask of the calling function when the current function
    // is popped.
    ++m_masked_exit_count;

    // Bumping the masked return count will cause the return mask(which is a
    // subset of the shader_mask) to be applied to the mask stack when
    // leaving if/else block.
    masked_function_context().return_count++;
}



void
LLVM_Util::op_masked_return()
{
    OSL_DEV_ONLY(std::cout << "push_mask_return" << std::endl);

    OSL_ASSERT(false == m_mask_stack.empty());

    const MaskInfo & mi = m_mask_stack.back();
    // Because we are inside a conditional branch we can't let our local
    // modified mask be directly used by other scopes, instead we must store
    // the result of to the stack for the outer scope to pick up and use.
    llvm::Value * loc_of_function_mask = masked_function_context().location_of_mask;
    llvm::Value * function_mask = op_load_mask(loc_of_function_mask);


    llvm::Value * return_from_mask = mi.mask;
    llvm::Value * modifiedMask;

    // For any active lanes of the mask we are returning from
    // set the function scope mask to 0.
    if (mi.negate) {
        modifiedMask = builder().CreateSelect(return_from_mask, function_mask, return_from_mask);
    } else {
        modifiedMask = builder().CreateSelect(return_from_mask, wide_constant_bool(false), function_mask);
    }

    op_store_mask(modifiedMask, loc_of_function_mask);

    masked_function_context().return_count++;
}



void
LLVM_Util::op_store (llvm::Value *val, llvm::Value *ptr)
{
    if(m_mask_stack.empty() || val->getType()->isVectorTy() == false || (!is_masking_required())) {

        //OSL_DEV_ONLY(std::cout << "unmasked op_store" << std::endl); We
        //may not be in a non-uniform code block or the value being stored
        //may be uniform, which case it shouldn't be a vector type
        builder().CreateStore (val, ptr);
    } else {
        //OSL_DEV_ONLY(std::cout << "MASKED op_store" << std::endl);
        // TODO: could probably make these OSL_DASSERT as  the conditional above "should" be checking all of this
        OSL_ASSERT(val->getType()->isVectorTy());
        OSL_ASSERT(false == m_mask_stack.empty());

        MaskInfo & mi = m_mask_stack.back();
        // TODO: add assert for ptr alignment in debug builds
#if 0
        if (m_supports_masked_stores) {
            builder().CreateMaskedStore(val, ptr, 64, mi.mask);
        } else
#endif
        {
            // Transform the masted store to a load+blend+store.
            // Technically, the behavior is different than a masked store as
            // different thread could technically have modified the masked
            // off data lane values inbetween the read+store. As this
            // language sits below the threading level that could never
            // happen and a read+.
            // TODO: Optimization, if we know this was the final store to
            // the ptr, we could force a masked store vs. load/blend
            llvm::Value *previous_value = builder().CreateLoad (ptr);
            if (false == mi.negate) {
                llvm::Value *blended_value = builder().CreateSelect(mi.mask, val, previous_value);
                builder().CreateStore(blended_value, ptr);
            } else {
                llvm::Value *blended_value = builder().CreateSelect(mi.mask, previous_value, val);
                builder().CreateStore(blended_value, ptr);
            }
        }
    }
}



void
LLVM_Util::op_unmasked_store (llvm::Value *val, llvm::Value *ptr)
{
    builder().CreateStore (val, ptr);
}



llvm::Value *
LLVM_Util::op_load_mask (llvm::Value *native_mask_ptr) {
    OSL_ASSERT(native_mask_ptr->getType() == type_ptr(type_native_mask()));

    return native_to_llvm_mask(op_load(native_mask_ptr));
}



void
LLVM_Util::op_store_mask (llvm::Value *llvm_mask, llvm::Value *native_mask_ptr)
{
    OSL_ASSERT(llvm_mask->getType() == type_wide_bool());
    OSL_ASSERT(native_mask_ptr->getType() == type_ptr(type_native_mask()));
    builder().CreateStore (llvm_mask_to_native(llvm_mask), native_mask_ptr);
}



llvm::Value *
LLVM_Util::GEP (llvm::Value *ptr, llvm::Value *elem)
{
    return builder().CreateGEP (ptr, elem);
}



llvm::Value *
LLVM_Util::GEP (llvm::Value *ptr, int elem)
{
    return builder().CreateConstGEP1_32 (ptr, elem);
}



llvm::Value *
LLVM_Util::GEP (llvm::Value *ptr, int elem1, int elem2)
{
    return builder().CreateConstGEP2_32 (nullptr, ptr, elem1, elem2);
}



llvm::Value *
LLVM_Util::op_add (llvm::Value *a, llvm::Value *b)
{
    if ((a->getType() == type_float() && b->getType() == type_float()) ||
        (a->getType() == type_wide_float() && b->getType() == type_wide_float()))
        return builder().CreateFAdd (a, b);
    if ((a->getType() == type_int() && b->getType() == type_int()) ||
        (a->getType() == type_wide_int() && b->getType() == type_wide_int()))
        return builder().CreateAdd (a, b);
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_sub (llvm::Value *a, llvm::Value *b)
{
    if ((a->getType() == type_float() && b->getType() == type_float()) ||
        (a->getType() == type_wide_float() && b->getType() == type_wide_float()))
        return builder().CreateFSub (a, b);
    if ((a->getType() == type_int() && b->getType() == type_int()) ||
        (a->getType() == type_wide_int() && b->getType() == type_wide_int()))
        return builder().CreateSub (a, b);
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_neg (llvm::Value *a)
{
    if ((a->getType() == type_float()) ||
        (a->getType() == type_wide_float()))
        return builder().CreateFNeg (a);
    if ((a->getType() == type_int()) ||
        (a->getType() == type_wide_int()))
        return builder().CreateNeg (a);
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_mul (llvm::Value *a, llvm::Value *b)
{
    if ((a->getType() == type_float() && b->getType() == type_float()) ||
        (a->getType() == type_wide_float() && b->getType() == type_wide_float()))
        return builder().CreateFMul (a, b);
    if ((a->getType() == type_int() && b->getType() == type_int()) ||
        (a->getType() == type_wide_int() && b->getType() == type_wide_int()))
        return builder().CreateMul (a, b);
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_div (llvm::Value *a, llvm::Value *b)
{
    if ((a->getType() == type_float() && b->getType() == type_float()) ||
        (a->getType() == type_wide_float() && b->getType() == type_wide_float()))
        return builder().CreateFDiv (a, b);
    if ((a->getType() == type_int() && b->getType() == type_int()) ||
        (a->getType() == type_wide_int() && b->getType() == type_wide_int()))
        return builder().CreateSDiv (a, b);
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_mod (llvm::Value *a, llvm::Value *b)
{
    if ((a->getType() == type_float() && b->getType() == type_float()) ||
        (a->getType() == type_wide_float() && b->getType() == type_wide_float()))
        return builder().CreateFRem (a, b);
    if ((a->getType() == type_int() && b->getType() == type_int()) ||
        (a->getType() == type_wide_int() && b->getType() == type_wide_int()))
        return builder().CreateSRem (a, b);

    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_float_to_int (llvm::Value* a)
{
    if (a->getType() == type_float())
        return builder().CreateFPToSI(a, type_int());
    if (a->getType() == type_wide_float())
        return builder().CreateFPToSI(a, type_wide_int());
    if ((a->getType() == type_int()) || (a->getType() == type_wide_int()))
        return a;
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_float_to_double (llvm::Value* a)
{
    if(a->getType() == type_float())
        return builder().CreateFPExt(a, type_double());
    if(a->getType() == type_wide_float())
        return builder().CreateFPExt(a, type_wide_double());
    // TODO: unclear why this is inconsistent vs. the other conversion ops
    // which become no-ops if the type is already the target
    
    OSL_DASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_int_to_longlong (llvm::Value* a)
{
    OSL_DASSERT (a->getType() == type_int());
    return builder().CreateSExt(a, llvm::Type::getInt64Ty(context()));
}


llvm::Value *
LLVM_Util::op_int_to_float (llvm::Value* a)
{
    if (a->getType() == type_int())
        return builder().CreateSIToFP(a, type_float());
    if (a->getType() == type_wide_int())
        return builder().CreateSIToFP(a, type_wide_float());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return a;
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_bool_to_int (llvm::Value* a)
{
    if (a->getType() == type_bool())
        return builder().CreateZExt (a, type_int());
    if (a->getType() == type_wide_bool()) 
        return builder().CreateZExt (a, type_wide_int());
    if ((a->getType() == type_int()) || (a->getType() == type_wide_int()))
        return a;
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}


llvm::Value *
LLVM_Util::op_bool_to_float (llvm::Value* a)
{
    if (a->getType() == type_bool())
        return builder().CreateSIToFP(a, type_float());
    if (a->getType() == type_wide_bool()) {
        return builder().CreateUIToFP(a, type_wide_float());
    }
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return a;
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}


llvm::Value *
LLVM_Util::op_int_to_bool(llvm::Value* a)
{
    if (a->getType() == type_int()) 
        return op_ne (a, constant(static_cast<int>(0)));
    if (a->getType() == type_wide_int()) 
        return op_ne (a, wide_constant(static_cast<int>(0)));
    if ((a->getType() == type_bool()) || (a->getType() == type_wide_bool()))
        return a;
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}


llvm::Value *
LLVM_Util::op_and (llvm::Value *a, llvm::Value *b)
{
    return builder().CreateAnd (a, b);
}


llvm::Value *
LLVM_Util::op_or (llvm::Value *a, llvm::Value *b)
{
    return builder().CreateOr (a, b);
}


llvm::Value *
LLVM_Util::op_xor (llvm::Value *a, llvm::Value *b)
{
    return builder().CreateXor (a, b);
}


llvm::Value *
LLVM_Util::op_shl (llvm::Value *a, llvm::Value *b)
{
    return builder().CreateShl (a, b);
}


llvm::Value *
LLVM_Util::op_shr (llvm::Value *a, llvm::Value *b)
{
    if ((a->getType() == type_int() && b->getType() == type_int()) ||
        (a->getType() == type_wide_int() && b->getType() == type_wide_int()))
        return builder().CreateAShr (a, b);  // signed int -> arithmetic shift
    
    OSL_ASSERT (0 && "Op has bad value type combination");
    return nullptr;
}



llvm::Value *
LLVM_Util::op_not (llvm::Value *a)
{
    return builder().CreateNot (a);
}



llvm::Value *
LLVM_Util::op_select (llvm::Value *cond, llvm::Value *a, llvm::Value *b)
{
    return builder().CreateSelect (cond, a, b);
}

llvm::Value *
LLVM_Util::op_zero_if(llvm::Value *cond, llvm::Value *v)
{
    OSL_ASSERT (v->getType() == type_wide_float() || v->getType() == type_wide_int() ||
                v->getType() == type_float() || v->getType() == type_int());

    bool is_wide = v->getType() == type_wide_float() || v->getType() == type_wide_int();
    bool is_float = v->getType() == type_float() || v->getType() == type_wide_float();
    llvm::Value * c_zero = (is_wide)?
                           (is_float) ? wide_constant(0.0f) : wide_constant(static_cast<int>(0))
                       :   (is_float) ? constant(0.0f) : constant(static_cast<int>(0));

    if (is_wide && m_supports_avx512f && ((m_vector_width == 8) || (m_vector_width == 16))) {
        // Select for AVX512 which repeats its parameter's operation with a mask.
        // However if that operation was used elsewhere we end up with 2 copies:
        // the original, and the masked version to implement the select.
        // The operation could be an expensive divide or sqrt!
        // Work is underway to fix this for LLVM 11.
        if (v->getNumUses() > 0) {

            // Workaround to avoid duplicating an expensive instruction:
            // insert an identity operation, to isolate original operation
            // from being duplicated.  In other words we are choosing to add an extra
            // inexpensive (0.5 clock) instruction rather than let something more expensive
            // be duplicated.
            // We can use a ternery log operation with a mask set to reproduce the 1st argument.
            llvm::Value *func = llvm::Intrinsic::getDeclaration(module(),
                    (m_vector_width == 16) ? llvm::Intrinsic::x86_avx512_pternlog_d_512
                                           : llvm::Intrinsic::x86_avx512_pternlog_d_256);
            //           (a, b, c) =  (111), (110), (101), (100), (011), (010), (001), (000)
            // a_identity(a, b, c) =    1      1      1      1      0      0      0      0    = 0xF0
            llvm::Value *a_identity_mask = constant(static_cast<int>(0xF0));
            llvm::Value *int_v = is_float ? builder().CreateBitCast(v, type_wide_int()) : v;
            llvm::Value *args[] = {int_v, int_v, int_v, a_identity_mask};
            llvm::Value *identity_call = builder().CreateCall(func, args);
            v = is_float ? builder().CreateBitCast(identity_call, type_wide_float()) : identity_call;
        }
    }
    return op_select (cond, c_zero, v);
}

llvm::Value *
LLVM_Util::op_extract (llvm::Value *a, int index)
{
    return builder().CreateExtractElement (a, index);
}

llvm::Value *
LLVM_Util::op_extract (llvm::Value *a, llvm::Value *index)
{
    return builder().CreateExtractElement (a, index);
}


llvm::Value *
LLVM_Util::op_insert (llvm::Value *v, llvm::Value *a, int index)
{
    return builder().CreateInsertElement (v, a, index);
}



llvm::Value *
LLVM_Util::op_eq (llvm::Value *a, llvm::Value *b, bool ordered)
{
    if (a->getType() != b->getType()) {
        std::cout << "a type=" << llvm_typenameof(a) << " b type=" << llvm_typenameof(b) << std::endl;
    }
    OSL_DASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOEQ (a, b) : builder().CreateFCmpUEQ (a, b);
    else
        return builder().CreateICmpEQ (a, b);
}



llvm::Value *
LLVM_Util::op_ne (llvm::Value *a, llvm::Value *b, bool ordered)
{
    OSL_DASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpONE (a, b) : builder().CreateFCmpUNE (a, b);
    else
        return builder().CreateICmpNE (a, b);
}



llvm::Value *
LLVM_Util::op_gt (llvm::Value *a, llvm::Value *b, bool ordered)
{
    OSL_DASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOGT (a, b) : builder().CreateFCmpUGT (a, b);
    else
        return builder().CreateICmpSGT (a, b);
}



llvm::Value *
LLVM_Util::op_lt (llvm::Value *a, llvm::Value *b, bool ordered)
{
    OSL_DASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOLT (a, b) : builder().CreateFCmpULT (a, b);
    else
        return builder().CreateICmpSLT (a, b);
}



llvm::Value *
LLVM_Util::op_ge (llvm::Value *a, llvm::Value *b, bool ordered)
{
    OSL_DASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOGE (a, b) : builder().CreateFCmpUGE (a, b);
    else
        return builder().CreateICmpSGE (a, b);
}



llvm::Value *
LLVM_Util::op_le (llvm::Value *a, llvm::Value *b, bool ordered)
{
    OSL_DASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOLE (a, b) : builder().CreateFCmpULE (a, b);
    else
        return builder().CreateICmpSLE (a, b);
}



llvm::Value *
LLVM_Util::op_fabs (llvm::Value *v)
{
    OSL_ASSERT (v->getType() == type_float() || v->getType() == type_wide_float());
    llvm::Type* types[] = { v->getType() };

    llvm::Function *func = llvm::Intrinsic::getDeclaration(module(), llvm::Intrinsic::fabs,
                                       types);

    llvm::Value *fabs_call = builder().CreateCall(func, { v });
    return fabs_call;
}

llvm::Value *
LLVM_Util::op_is_not_finite (llvm::Value *v)
{
    OSL_ASSERT (v->getType() == type_float() || v->getType() == type_wide_float());

    if (m_supports_avx512f && v->getType() == type_wide_float()) {
        OSL_ASSERT((m_vector_width == 8) || (m_vector_width == 16));
        llvm::Value *func = llvm::Intrinsic::getDeclaration(module(),
            (v->getType() == type_wide_float()) ?
                    ((m_vector_width == 16) ? llvm::Intrinsic::x86_avx512_fpclass_ps_512 : llvm::Intrinsic::x86_avx512_fpclass_ps_256)
                    : llvm::Intrinsic::x86_avx512_mask_fpclass_ss);
        // Flags:
        //        0x01 // QNaN
        //        0x02 // Positive Zero
        //        0x04 // Negative Zero
        //        0x08 // Positive Infinity
        //        0x10 // Negative Infinity
        //        0x20 // Denormal
        //        0x40 // Negative
        //        0x80 // SNaN

        llvm::Value *flags = constant(static_cast<int>(0x01 /*QNaN*/ |
                                                       0x08 /*Positive Infinity*/ |
                                                       0x10 /*Negative Infinity*/));
        llvm::Value *args[] = {v, flags};
        llvm::Value *maskOfNonFiniteValues = call_function(func, args);
        return maskOfNonFiniteValues;
    } else {
        llvm::Value *fabs_v = op_fabs(v);
        llvm::Constant *infinity = llvm::ConstantFP::getInfinity(v->getType());
        // FCMP_ONE: yields true if both operands are not a QNAN and op1 is not equal to op2.
        // So if v == QNAN, we expect result of FCMP_ONE to always be false
        // otherwise will return false when v == +infinity, which covers -infinity, +infinity, and QNAN
        llvm::Value *result = builder().CreateFCmp(llvm::CmpInst::FCMP_ONE, fabs_v, infinity, "ordered equals infinity");
        return op_not(result);
    }
}


void
LLVM_Util::write_bitcode_file (const char *filename, std::string *err)
{
    std::error_code local_error;
    llvm::raw_fd_ostream out (filename, local_error, llvm::sys::fs::F_None);
    if (! out.has_error()) {
        llvm::WriteBitcodeToFile (*module(), out);
        if (err && local_error)
            *err = local_error.message ();
    }
}



bool
LLVM_Util::ptx_compile_group (llvm::Module* lib_module, const std::string& name,
                              std::string& out)
{
    std::string target_triple = module()->getTargetTriple();

    OSL_ASSERT (lib_module == nullptr ||
                (lib_module->getTargetTriple() == target_triple &&
                 "PTX compile error: Shader and renderer bitcode library targets do not match"));

    // Create a new empty module to hold the linked shadeops and compiled
    // ShaderGroup
    llvm::Module* linked_module = new_module (name.c_str());

    // First, link in the cloned ShaderGroup module
    std::unique_ptr<llvm::Module> mod_ptr = llvm::CloneModule (*module());
    bool failed = llvm::Linker::linkModules (*linked_module, std::move (mod_ptr));
    OSL_ASSERT (!failed && "PTX compile error: Unable to link group module");

    // Second, link in the shadeops library, keeping only the functions that are needed

    if (lib_module) {
        std::unique_ptr<llvm::Module> lib_ptr (lib_module);
        failed = llvm::Linker::linkModules (*linked_module, std::move (lib_ptr));
    }

#if (OPTIX_VERSION < 70000)
    // Internalize the Globals to match code generated by NVCC
    for (auto& g_var : linked_module->globals()) {
        g_var.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
#endif

    // Verify that the NVPTX target has been initialized
    std::string error;
    const llvm::Target* llvm_target =
        llvm::TargetRegistry::lookupTarget (target_triple, error);
    OSL_ASSERT (llvm_target && "PTX compile error: LLVM Target is not initialized");

    llvm::TargetOptions  options;
    options.AllowFPOpFusion = llvm::FPOpFusion::Standard;
    // N.B. 'Standard' only allow fusion of 'blessed' ops (currently just
    // fmuladd). To truly disable FMA and never fuse FP-ops, we need to
    // instead use llvm::FPOpFusion::Strict.
    options.UnsafeFPMath                           = 1;
    options.NoInfsFPMath                           = 1;
    options.NoNaNsFPMath                           = 1;
    options.HonorSignDependentRoundingFPMathOption = 0;
    options.FloatABIType                           = llvm::FloatABI::Default;
    options.AllowFPOpFusion                        = llvm::FPOpFusion::Fast;
    options.NoZerosInBSS                           = 0;
    options.GuaranteedTailCallOpt                  = 0;
    options.StackAlignmentOverride                 = 0;
    options.UseInitArray                           = 0;

    llvm::TargetMachine* target_machine = llvm_target->createTargetMachine(
        target_triple, CUDA_TARGET_ARCH, "+ptx50", options,
        llvm::Reloc::Static, llvm::CodeModel::Small, llvm::CodeGenOpt::Aggressive);
    OSL_ASSERT (target_machine && "PTX compile error: Unable to create target machine -- is NVPTX enabled in LLVM?");

    // Setup the optimization passes
    llvm::legacy::FunctionPassManager fn_pm (linked_module);
    fn_pm.add (llvm::createTargetTransformInfoWrapperPass (
                   target_machine->getTargetIRAnalysis()));

    llvm::legacy::PassManager mod_pm;
    mod_pm.add (new llvm::TargetLibraryInfoWrapperPass (llvm::Triple (target_triple)));
    mod_pm.add (llvm::createTargetTransformInfoWrapperPass (
                    target_machine->getTargetIRAnalysis()));
    mod_pm.add (llvm::createRewriteSymbolsPass());

    // Make sure the 'flush-to-zero' instruction variants are used when possible
    linked_module->addModuleFlag (llvm::Module::Override, "nvvm-reflect-ftz", 1);
    for (llvm::Function& fn : *linked_module) {
        fn.addFnAttr ("nvptx-f32ftz", "true");
    }

    llvm::SmallString<4096>   assembly;
    llvm::raw_svector_ostream assembly_stream (assembly);

    // TODO: Make sure rounding modes, etc., are set correctly
#if OSL_LLVM_VERSION >= 100
    target_machine->addPassesToEmitFile (mod_pm, assembly_stream,
                                         nullptr,  // FIXME: Correct?
                                         llvm::CGFT_AssemblyFile);
#else
    target_machine->addPassesToEmitFile (mod_pm, assembly_stream,
                                         nullptr,  // FIXME: Correct?
                                         llvm::TargetMachine::CGFT_AssemblyFile);
#endif

    // Run the optimization passes on the functions
    fn_pm.doInitialization();
    for (llvm::Module::iterator i = linked_module->begin(); i != linked_module->end(); i++) {
        fn_pm.run (*i);
    }

    // Run the optimization passes on the module to generate the PTX
    mod_pm.run (*linked_module);

    // TODO: Minimize string copying
    out = assembly_stream.str().str();

    delete linked_module;

    return true;
}



std::string
LLVM_Util::bitcode_string (llvm::Function *func)
{
    std::string s;
    llvm::raw_string_ostream stream (s);
    stream << (*func);
    return stream.str();
}



std::string
LLVM_Util::bitcode_string (llvm::Module *module)
{
    std::string s;
    llvm::raw_string_ostream stream (s);

    for (auto&& func : module->getFunctionList())
        stream << func << '\n';

    return stream.str();
}



std::string
LLVM_Util::module_string()
{
    std::string s;
    llvm::raw_string_ostream stream(s);
    m_llvm_module->print(stream, nullptr);
    return s;
}



void
LLVM_Util::delete_func_body (llvm::Function *func)
{
    func->deleteBody ();
}



bool
LLVM_Util::func_is_empty (llvm::Function *func)
{
    return func->size() == 1 // func has just one basic block
        && func->front().size() == 1;  // the block has one instruction,
                                       ///   presumably the ret
}


std::string
LLVM_Util::func_name (llvm::Function *func)
{
    return func->getName().str();
}



llvm::DIFile *
LLVM_Util::getOrCreateDebugFileFor(const std::string &file_name)
{
    auto iter = mDebugFileByName.find(file_name);
    if (iter == mDebugFileByName.end()) {
        //OSL_DEV_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>CREATING FILE<<<<<<<<<<<<<<<<<<<<<<<<< " << file_name << std::endl);
        OSL_ASSERT(m_llvm_debug_builder != nullptr);
        llvm::DIFile *file = m_llvm_debug_builder->createFile(
                file_name, ".\\");
        mDebugFileByName.insert(std::make_pair(file_name,file));
        return file;
    }
    return iter->second;
}



llvm::DIScope *
LLVM_Util::getCurrentDebugScope() const
{
    OSL_ASSERT(mDebugCU != nullptr);
    if (mLexicalBlocks.empty()) {
        return mDebugCU;
    } else {
        return mLexicalBlocks.back();
    }
}



llvm::DILocation *
LLVM_Util::getCurrentInliningSite() const
{
    if (mInliningSites.empty()) {
        return nullptr;
    } else {
        return mInliningSites.back();
    }
}

}; // namespace pvt
OSL_NAMESPACE_EXIT
