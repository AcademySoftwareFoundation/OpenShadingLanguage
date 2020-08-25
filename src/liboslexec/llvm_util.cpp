// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include <memory>
#include <cinttypes>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/thread.h>
#include <boost/thread/tss.hpp>   /* for thread_specific_ptr */

#include <OSL/oslconfig.h>
#include <OSL/llvm_util.h>

#if OSL_LLVM_VERSION < 70
#error "LLVM minimum version required for OSL is 7.0"
#endif

#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
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
#include <llvm/IR/Verifier.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>

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

static inline llvm::VectorType* llvmVectorGet(llvm::Type *llvmType, unsigned width) {
#if OSL_LLVM_VERSION < 110
    return llvm::VectorType::get(llvmType, width);
#else
    return llvm::VectorType::get(llvmType, width, false);
#endif
}

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
class LLVM_Util::MemoryManager : public LLVMMemoryManager {
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



class LLVM_Util::IRBuilder : public llvm::IRBuilder<llvm::ConstantFolder,
                                               llvm::IRBuilderDefaultInserter> {
    typedef llvm::IRBuilder<llvm::ConstantFolder,
                            llvm::IRBuilderDefaultInserter> Base;
public:
    IRBuilder(llvm::BasicBlock *TheBB) : Base(TheBB) {}
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
      mVTuneNotifier(nullptr),
      m_llvm_debug_builder(nullptr),
      mDebugCU(nullptr),
      mSubTypeForInlinedFunction(nullptr),
      m_ModuleIsFinalized(false),
      m_ModuleIsPruned(false)
{
    SetupLLVM ();
    m_thread = per_thread_info.get();
    OSL_ASSERT (m_thread);

    {
        OIIO::spin_lock lock (llvm_global_mutex);
        if (! m_thread->llvm_context)
            m_thread->llvm_context = new llvm::LLVMContext();

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
    m_llvm_type_wide_float = llvmVectorGet(m_llvm_type_float, m_vector_width);
    m_llvm_type_wide_double = llvmVectorGet(m_llvm_type_double, m_vector_width);
    m_llvm_type_wide_int = llvmVectorGet(m_llvm_type_int, m_vector_width);
    m_llvm_type_wide_bool = llvmVectorGet(m_llvm_type_bool, m_vector_width);
    m_llvm_type_wide_char = llvmVectorGet(m_llvm_type_char, m_vector_width);
    m_llvm_type_wide_longlong = llvmVectorGet(m_llvm_type_longlong, m_vector_width);
    
    m_llvm_type_wide_char_ptr = llvm::PointerType::get(m_llvm_type_wide_char, 0);
    m_llvm_type_wide_ustring_ptr = llvmVectorGet(m_llvm_type_char_ptr, m_vector_width);
    m_llvm_type_wide_void_ptr = llvmVectorGet(m_llvm_type_void_ptr, m_vector_width);
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
                : llvmVectorGet(m_llvm_type_int, m_vector_width);

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

    if (target_host) {
        llvm::TargetMachine* target_machine = execengine()->getTargetMachine();
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
    // Tests on production shaders suggest the sweet spot
    // between JIT time and runtime performance is O1.
    llvm::PassManagerBuilder builder;
    builder.OptLevel = optlevel;
    builder.Inliner = llvm::createFunctionInliningPass();
    // builder.DisableUnrollLoops = true;
    builder.populateFunctionPassManager (fpm);
    builder.populateModulePassManager (mpm);
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

llvm::Value *
LLVM_Util::wide_constant (llvm::Value * constant_val)
{
    llvm::Constant *cv = llvm::dyn_cast<llvm::Constant>(constant_val);
    OSL_ASSERT(cv  != nullptr);
    return llvm::ConstantDataVector::getSplat(m_vector_width, cv);
}


llvm::Value *
LLVM_Util::constant (float f)
{
    return llvm::ConstantFP::get (context(), llvm::APFloat(f));
}

llvm::Value *
LLVM_Util::wide_constant (float f)
{
    return llvm::ConstantDataVector::getSplat(m_vector_width, llvm::ConstantFP::get (context(), llvm::APFloat(f)));
}

llvm::Value *
LLVM_Util::constant (int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(32,i));
}


llvm::Value *
LLVM_Util::constant8 (int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(8,i));
}

llvm::Value *
LLVM_Util::constant16 (uint16_t i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(16,i));
}

llvm::Value *
LLVM_Util::constant64 (uint64_t i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(64,i));
}

llvm::Value *
LLVM_Util::constant128 (uint64_t i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(128,i));
}

llvm::Value *
LLVM_Util::constant128 (uint64_t left, uint64_t right)
{
    uint64_t bigNum[2];
    bigNum[0] = left;
    bigNum[1] = right;
    llvm::ArrayRef< uint64_t > refBigNum(&bigNum[0], 2);
    return llvm::ConstantInt::get (context(), llvm::APInt(128,refBigNum));
}


llvm::Value *
LLVM_Util::wide_constant (int i)
{
    return llvm::ConstantDataVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(32,i)));
}

llvm::Value *
LLVM_Util::constant (size_t i)
{
    int bits = sizeof(size_t)*8;
    return llvm::ConstantInt::get (context(), llvm::APInt(bits,i));
}

llvm::Value *
LLVM_Util::wide_constant (size_t i)
{
    int bits = sizeof(size_t)*8;
    return llvm::ConstantDataVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(bits,i)));
}

llvm::Value *
LLVM_Util::constant_bool (bool i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(1,i));
}

llvm::Value *
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


llvm::Value *
LLVM_Util::constant (const TypeDesc &type)
{
    long long *i = (long long *)&type;
    return llvm::ConstantInt::get (context(), llvm::APInt(64,*i));
}



llvm::Value *
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



void
LLVM_Util::op_store (llvm::Value *val, llvm::Value *ptr)
{
    builder().CreateStore (val, ptr);
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
        target_triple, "sm_35", "+ptx50", options,
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
