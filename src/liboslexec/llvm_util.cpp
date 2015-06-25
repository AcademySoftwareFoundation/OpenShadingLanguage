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


#include <OpenImageIO/thread.h>

#include "OSL/oslconfig.h"
#include "OSL/llvm_util.h"

#if OSL_LLVM_VERSION >= 35 && ! OSL_BUILD_CPP11
#error "LLVM >= 3.5 requires USE_CPP11=1"
#endif

#ifndef USE_MCJIT
  // MCJIT first appeared with LLVM 3.3
# define USE_MCJIT (OSL_LLVM_VERSION>=33)
#endif

// MCJIT is mandatory for LLVM 3.6 and beyond, no more old JIT
#define MCJIT_REQUIRED (USE_MCJIT >= 2 || OSL_LLVM_VERSION >= 36)

#if MCJIT_REQUIRED
# undef USE_MCJIT
# define USE_MCJIT 2
#endif

#if OSL_LLVM_VERSION >= 33

# include <llvm/IR/Constants.h>
# include <llvm/IR/DerivedTypes.h>
# include <llvm/IR/Instructions.h>
# include <llvm/IR/Intrinsics.h>
# include <llvm/IR/Module.h>
# include <llvm/IR/LLVMContext.h>
# include <llvm/IR/IRBuilder.h>
# include <llvm/IR/DataLayout.h>
# if OSL_LLVM_VERSION >= 35
#   include <llvm/Linker/Linker.h>
#   include <llvm/Support/FileSystem.h>
# else
#   include <llvm/Linker.h>
# endif
# if OSL_LLVM_VERSION >= 34
#   include <llvm/Support/ErrorOr.h>
#   include <llvm/IR/LegacyPassManager.h>
# else
#   include <llvm/PassManager.h>
# endif
# include <llvm/Support/TargetRegistry.h>

#else /* older releases */

# include <llvm/Constants.h>
# include <llvm/DerivedTypes.h>
# include <llvm/Instructions.h>
# include <llvm/Intrinsics.h>
# include <llvm/Linker.h>
# include <llvm/LLVMContext.h>
# include <llvm/Module.h>
# if OSL_LLVM_VERSION == 32
#   include <llvm/IRBuilder.h>
#   include <llvm/DataLayout.h>
# else /* older releases */
#   include <llvm/Support/IRBuilder.h>
#   include <llvm/Target/TargetData.h>
# endif
# include <llvm/PassManager.h>

#endif

#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#if USE_MCJIT
# include <llvm/ExecutionEngine/MCJIT.h>
#endif
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/ExecutionEngine/JITMemoryManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/PrettyStackTrace.h>
#if OSL_LLVM_VERSION >= 35
#include <llvm/IR/Verifier.h>
#else
#include <llvm/Analysis/Verifier.h>
#endif
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

OSL_NAMESPACE_ENTER

namespace pvt {


namespace {
static OIIO::spin_mutex llvm_global_mutex;
static bool setup_done = false;
static OIIO::thread_specific_ptr<LLVM_Util::PerThreadInfo> perthread_infos;
static std::vector<shared_ptr<llvm::JITMemoryManager> > jitmm_hold;
};




// We hold certain things (LLVM context and custom JIT memory manager)
// per thread and retained across LLVM_Util invocations.  We are
// intentionally "leaking" them.
struct LLVM_Util::PerThreadInfo {
    PerThreadInfo () : llvm_context(NULL), llvm_jitmm(NULL) {}
    ~PerThreadInfo () {
        delete llvm_context;
        // N.B. Do NOT delete the jitmm -- another thread may need the
        // code! Don't worry, we stashed a pointer in jitmm_hold.
    }
    static void destroy (PerThreadInfo *threadinfo) { delete threadinfo; }
    static PerThreadInfo *get () {
        PerThreadInfo *p = perthread_infos.get ();
        if (! p) {
            p = new PerThreadInfo();
            perthread_infos.reset (p);
        }
        return p;
    }

    llvm::LLVMContext *llvm_context;
    llvm::JITMemoryManager *llvm_jitmm;
};




size_t
LLVM_Util::total_jit_memory_held ()
{
    size_t jitmem = 0;
    OIIO::spin_lock lock (llvm_global_mutex);
    for (size_t i = 0;  i < jitmm_hold.size();  ++i) {
        llvm::JITMemoryManager *mm = jitmm_hold[i].get();
        if (mm)
            jitmem += mm->GetDefaultCodeSlabSize() * mm->GetNumCodeSlabs()
                    + mm->GetDefaultDataSlabSize() * mm->GetNumDataSlabs()
                    + mm->GetDefaultStubSlabSize() * mm->GetNumStubSlabs();
    }
    return jitmem;
}



/// OSL_Dummy_JITMemoryManager - Create a shell that passes on requests
/// to a real JITMemoryManager underneath, but can be retained after the
/// dummy is destroyed.  Also, we don't pass along any deallocations.
class OSL_Dummy_JITMemoryManager : public llvm::JITMemoryManager {
protected:
    llvm::JITMemoryManager *mm;  // the real one
public:
    OSL_Dummy_JITMemoryManager(llvm::JITMemoryManager *realmm) : mm(realmm) { HasGOT = realmm->isManagingGOT(); }
    virtual ~OSL_Dummy_JITMemoryManager() {}
    virtual void setMemoryWritable() { mm->setMemoryWritable(); }
    virtual void setMemoryExecutable() { mm->setMemoryExecutable(); }
    virtual void setPoisonMemory(bool poison) { mm->setPoisonMemory(poison); }
    virtual void AllocateGOT() { ASSERT(HasGOT == false); ASSERT(HasGOT == mm->isManagingGOT()); mm->AllocateGOT(); HasGOT = true; ASSERT(HasGOT == mm->isManagingGOT()); }
    virtual uint8_t *getGOTBase() const { return mm->getGOTBase(); }
    virtual uint8_t *startFunctionBody(const llvm::Function *F,
                                       uintptr_t &ActualSize) {
        return mm->startFunctionBody (F, ActualSize);
    }
    virtual uint8_t *allocateStub(const llvm::GlobalValue* F, unsigned StubSize,
                                  unsigned Alignment) {
        return mm->allocateStub (F, StubSize, Alignment);
    }
    virtual void endFunctionBody(const llvm::Function *F,
                                 uint8_t *FunctionStart, uint8_t *FunctionEnd) {
        mm->endFunctionBody (F, FunctionStart, FunctionEnd);
    }
    virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment) {
        return mm->allocateSpace (Size, Alignment);
    }
    virtual uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment) {
        return mm->allocateGlobal (Size, Alignment);
    }
    virtual void deallocateFunctionBody(void *Body) {
        // DON'T DEALLOCATE mm->deallocateFunctionBody (Body);
    }
#if OSL_LLVM_VERSION <= 33
    virtual uint8_t* startExceptionTable(const llvm::Function* F,
                                         uintptr_t &ActualSize) {
        return mm->startExceptionTable (F, ActualSize);
    }
    virtual void endExceptionTable(const llvm::Function *F, uint8_t *TableStart,
                                   uint8_t *TableEnd, uint8_t* FrameRegister) {
        mm->endExceptionTable (F, TableStart, TableEnd, FrameRegister);
    }
    virtual void deallocateExceptionTable(void *ET) {
        // DON'T DEALLOCATE mm->deallocateExceptionTable(ET);
    }
#endif
    virtual bool CheckInvariants(std::string &s) {
        return mm->CheckInvariants(s);
    }
    virtual size_t GetDefaultCodeSlabSize() {
        return mm->GetDefaultCodeSlabSize();
    }
    virtual size_t GetDefaultDataSlabSize() {
        return mm->GetDefaultDataSlabSize();
    }
    virtual size_t GetDefaultStubSlabSize() {
        return mm->GetDefaultStubSlabSize();
    }
    virtual unsigned GetNumCodeSlabs() { return mm->GetNumCodeSlabs(); }
    virtual unsigned GetNumDataSlabs() { return mm->GetNumDataSlabs(); }
    virtual unsigned GetNumStubSlabs() { return mm->GetNumStubSlabs(); }

#if OSL_LLVM_VERSION >= 34

    virtual void *getPointerToNamedFunction(const std::string &Name,
                                            bool AbortOnFailure = true) {
        return mm->getPointerToNamedFunction (Name, AbortOnFailure);
    }
    virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                             unsigned SectionID, llvm::StringRef SectionName) {
        return mm->allocateCodeSection(Size, Alignment, SectionID, SectionName);
    }
    virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                             unsigned SectionID, llvm::StringRef SectionName,
                             bool IsReadOnly) {
        return mm->allocateDataSection(Size, Alignment, SectionID,
                                       SectionName, IsReadOnly);
    }
    virtual void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) {
        mm->registerEHFrames (Addr, LoadAddr, Size);
    }
    virtual void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) {
        mm->deregisterEHFrames(Addr, LoadAddr, Size);
    }
    virtual uint64_t getSymbolAddress(const std::string &Name) {
        return mm->getSymbolAddress (Name);
    }
    virtual void notifyObjectLoaded(llvm::ExecutionEngine *EE, const llvm::ObjectImage *oi) {
        mm->notifyObjectLoaded (EE, oi);
    }
    virtual bool finalizeMemory(std::string *ErrMsg = 0) {
        return mm->finalizeMemory (ErrMsg);
    }

#elif OSL_LLVM_VERSION == 33

    virtual void *getPointerToNamedFunction(const std::string &Name,
                                            bool AbortOnFailure = true) {
        return mm->getPointerToNamedFunction (Name, AbortOnFailure);
    }
    virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID) {
        return mm->allocateCodeSection(Size, Alignment, SectionID);
    }
    virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID, bool IsReadOnly) {
        return mm->allocateDataSection(Size, Alignment, SectionID, IsReadOnly);
    }
    virtual bool applyPermissions(std::string *ErrMsg = 0) {
        return mm->applyPermissions(ErrMsg);
    }

#elif OSL_LLVM_VERSION == 32 || OSL_LLVM_VERSION == 31

    virtual void *getPointerToNamedFunction(const std::string &Name,
                                            bool AbortOnFailure = true) {
        return mm->getPointerToNamedFunction (Name, AbortOnFailure);
    }
    virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID) {
        return mm->allocateCodeSection(Size, Alignment, SectionID);
    }
    virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID) {
        return mm->allocateDataSection(Size, Alignment, SectionID);
    }

#endif
};




LLVM_Util::LLVM_Util (int debuglevel)
    : m_debug(debuglevel), m_mcjit(MCJIT_REQUIRED), m_thread(NULL),
      m_llvm_context(NULL), m_llvm_module(NULL),
      m_builder(NULL), m_llvm_jitmm(NULL),
      m_current_function(NULL),
      m_llvm_module_passes(NULL), m_llvm_func_passes(NULL),
      m_llvm_exec(NULL)
{
    SetupLLVM ();
    m_thread = PerThreadInfo::get();
    ASSERT (m_thread);

    {
        OIIO::spin_lock lock (llvm_global_mutex);
        if (! m_thread->llvm_context)
            m_thread->llvm_context = new llvm::LLVMContext();

        if (! m_thread->llvm_jitmm) {
            m_thread->llvm_jitmm = llvm::JITMemoryManager::CreateDefaultMemManager();
            ASSERT (m_thread->llvm_jitmm);
            jitmm_hold.push_back (shared_ptr<llvm::JITMemoryManager>(m_thread->llvm_jitmm));
        }
    }

    m_llvm_context = m_thread->llvm_context;
    m_llvm_jitmm = new OSL_Dummy_JITMemoryManager(m_thread->llvm_jitmm);

    // Set up aliases for types we use over and over
    m_llvm_type_float = (llvm::Type *) llvm::Type::getFloatTy (*m_llvm_context);
    m_llvm_type_int = (llvm::Type *) llvm::Type::getInt32Ty (*m_llvm_context);
    if (sizeof(char *) == 4)
        m_llvm_type_addrint = (llvm::Type *) llvm::Type::getInt32Ty (*m_llvm_context);
    else
        m_llvm_type_addrint = (llvm::Type *) llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_int_ptr = (llvm::PointerType *) llvm::Type::getInt32PtrTy (*m_llvm_context);
    m_llvm_type_bool = (llvm::Type *) llvm::Type::getInt1Ty (*m_llvm_context);
    m_llvm_type_char = (llvm::Type *) llvm::Type::getInt8Ty (*m_llvm_context);
    m_llvm_type_longlong = (llvm::Type *) llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_void = (llvm::Type *) llvm::Type::getVoidTy (*m_llvm_context);
    m_llvm_type_char_ptr = (llvm::PointerType *) llvm::Type::getInt8PtrTy (*m_llvm_context);
    m_llvm_type_float_ptr = (llvm::PointerType *) llvm::Type::getFloatPtrTy (*m_llvm_context);
    m_llvm_type_ustring_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_char_ptr, 0);
    m_llvm_type_void_ptr = m_llvm_type_char_ptr;

    // A triple is a struct composed of 3 floats
    std::vector<llvm::Type*> triplefields(3, m_llvm_type_float);
    m_llvm_type_triple = type_struct (triplefields, "Vec3");
    m_llvm_type_triple_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_triple, 0);

    // A matrix is a struct composed 16 floats
    std::vector<llvm::Type*> matrixfields(16, m_llvm_type_float);
    m_llvm_type_matrix = type_struct (matrixfields, "Matrix4");
    m_llvm_type_matrix_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_matrix, 0);
}



LLVM_Util::~LLVM_Util ()
{
    execengine (NULL);
    delete m_llvm_module_passes;
    delete m_llvm_func_passes;
    delete m_builder;
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

#if OSL_LLVM_VERSION <= 33
    // Starting with LLVM 3.4, the pretty stack trace was opt-in rather
    // than opt-out, and the following variable was removed.
    llvm::DisablePrettyStackTrace = true;
#endif

#if OSL_LLVM_VERSION < 35
    // enable it to be thread-safe
    llvm::llvm_start_multithreaded ();
#endif
// new versions (>=3.5)don't need this anymore

#if USE_MCJIT
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllDisassemblers();
#else
    llvm::InitializeNativeTarget();
#endif

    if (debug()) {
#if OSL_LLVM_VERSION >= 33
        for (llvm::TargetRegistry::iterator t = llvm::TargetRegistry::begin();
             t != llvm::TargetRegistry::end();  ++t) {
            std::cout << "Target: '" << t->getName() << "' "
                      << t->getShortDescription() << "\n";
        }
        std::cout << "\n";
#endif
    }

    setup_done = true;
}



llvm::Module *
LLVM_Util::new_module (const char *id)
{
    return new llvm::Module(id, context());
}



llvm::Module *
LLVM_Util::module_from_bitcode (const char *bitcode, size_t size,
                                const std::string &name, std::string *err)
{
    if (err)
        err->clear();

#if OSL_LLVM_VERSION >= 36
    llvm::MemoryBufferRef buf =
        llvm::MemoryBufferRef(llvm::StringRef(bitcode, size), name));
#else /* LLVM 3.5 or earlier */
    llvm::MemoryBuffer* buf =
        llvm::MemoryBuffer::getMemBuffer (llvm::StringRef(bitcode, size), name);
#endif

    // Load the LLVM bitcode and parse it into a Module
    llvm::Module *m = NULL;

#if USE_MCJIT /* Parse the whole thing now */
    if (mcjit() || MCJIT_REQUIRED) {
        // FIXME!! Using MCJIT should not require unconditionally parsing
        // the bitcode. But for now, when using getLazyBitcodeModule to
        // lazily deserialize the bitcode, MCJIT is unable to find the
        // called functions due to disagreement about whether a leading "_"
        // is part of the symbol name.
  #if OSL_LLVM_VERSION >= 35
        llvm::ErrorOr<llvm::Module *> ModuleOrErr = llvm::parseBitcodeFile (buf, context());
        if (std::error_code EC = ModuleOrErr.getError())
            if (err)
              *err = EC.message();
        m = ModuleOrErr.get();
  #else
        m = llvm::ParseBitcodeFile (buf, context(), err);
  #endif
  #if OSL_LLVM_VERSION < 36
        delete buf;
  #endif
    }
    else
#endif
    {
        // Create a lazily deserialized IR module
        // This can only be done for old JIT
# if OSL_LLVM_VERSION >= 35
        m = llvm::getLazyBitcodeModule (buf, context()).get();
# else
        m = llvm::getLazyBitcodeModule (buf, context(), err);
# endif
        // don't delete buf, the module has taken ownership of it
    }

    // Debugging: print all functions in the module
    // for (llvm::Module::iterator i = m->begin(); i != m->end(); ++i)
    //     std::cout << "  found " << i->getName().data() << "\n";
    return m;
}



void
LLVM_Util::new_builder (llvm::BasicBlock *block)
{
    end_builder();
    if (! block)
        block = new_basic_block ();
    m_builder = new llvm::IRBuilder<> (block);
}



void
LLVM_Util::end_builder ()
{
    delete m_builder;
    m_builder = NULL;
}



llvm::ExecutionEngine *
LLVM_Util::make_jit_execengine (std::string *err)
{
    execengine (NULL);   // delete and clear any existing engine
    if (err)
        err->clear ();
#if OSL_LLVM_VERSION >= 33
    m_llvm_exec = llvm::EngineBuilder(module())
                            .setEngineKind(llvm::EngineKind::JIT)
                            .setErrorStr(err)
                            .setJITMemoryManager(jitmm())
                            .setOptLevel(llvm::CodeGenOpt::Default)
                            .setUseMCJIT(mcjit() || MCJIT_REQUIRED)
                            .create();
#else
    m_llvm_exec = llvm::ExecutionEngine::createJIT (module(), err,
                                    jitmm(), llvm::CodeGenOpt::Default,
                                    /*AllocateGVsWithCode*/ false);
#endif

    // N.B. createJIT will take ownership of the the JITMemoryManager!

    if (! m_llvm_exec)
        return NULL;

    // Force it to JIT as soon as we ask it for the code pointer,
    // don't take any chances that it might JIT lazily, since we
    // will be stealing the JIT code memory from under its nose and
    // destroying the Module & ExecutionEngine.
    m_llvm_exec->DisableLazyCompilation ();
    return m_llvm_exec;
}



void
LLVM_Util::execengine (llvm::ExecutionEngine *exec)
{
    delete m_llvm_exec;
    m_llvm_exec = exec;
}



void *
LLVM_Util::getPointerToFunction (llvm::Function *func)
{
    llvm::ExecutionEngine *exec = execengine();
#if OSL_LLVM_VERSION >= 33
    if (USE_MCJIT)
        exec->finalizeObject ();
#endif
    void *f = exec->getPointerToFunction (func);
    ASSERT (f && "could not getPointerToFunction");
    return f;
}



void
LLVM_Util::InstallLazyFunctionCreator (void* (*P)(const std::string &))
{
    llvm::ExecutionEngine *exec = execengine();
    exec->InstallLazyFunctionCreator (P);
}



void
LLVM_Util::setup_optimization_passes (int optlevel)
{
    ASSERT (m_llvm_module_passes == NULL && m_llvm_func_passes == NULL);

    // Specify per-function passes
    //
#if OSL_LLVM_VERSION >= 34
    m_llvm_func_passes = new llvm::legacy::FunctionPassManager(module());
    llvm::legacy::FunctionPassManager &fpm (*m_llvm_func_passes);
# if OSL_LLVM_VERSION >= 35
    fpm.add (new llvm::DataLayoutPass(module()));
# else
    fpm.add (new llvm::DataLayout(module()));
# endif
#else
    m_llvm_func_passes = new llvm::FunctionPassManager(module());
    llvm::FunctionPassManager &fpm (*m_llvm_func_passes);
# if OSL_LLVM_VERSION >= 32
    fpm.add (new llvm::DataLayout(module()));
# else
    fpm.add (new llvm::TargetData(module()));
# endif
#endif

    // Specify module-wide (interprocedural optimization) passes
    //
#if OSL_LLVM_VERSION >= 34
    m_llvm_module_passes = new llvm::legacy::PassManager;
    llvm::legacy::PassManager &mpm (*m_llvm_module_passes);
# if OSL_LLVM_VERSION >= 35
    mpm.add (new llvm::DataLayoutPass(module()));
# else
    mpm.add (new llvm::DataLayout(module()));
# endif
#else
    m_llvm_module_passes = new llvm::PassManager;
    llvm::PassManager &mpm (*m_llvm_module_passes);
#if OSL_LLVM_VERSION >= 32
    mpm.add (new llvm::DataLayout(module()));
#else
    mpm.add (new llvm::TargetData(module()));
#endif
#endif

    if (optlevel >= 1 && optlevel <= 3) {
#if OSL_LLVM_VERSION <= 34
        // For LLVM 3.0 and higher, llvm_optimize 1-3 means to use the
        // same set of optimizations as clang -O1, -O2, -O3
        llvm::PassManagerBuilder builder;
        builder.OptLevel = optlevel;
        builder.Inliner = llvm::createFunctionInliningPass();
        // builder.DisableUnrollLoops = true;
        builder.populateFunctionPassManager (fpm);
        builder.populateModulePassManager (mpm);
#endif

    } else {
        // LLVM 2.x, or unknown choices for llvm_optimize: use the same basic
        // set of passes that we always have.

        // Always add verifier?
        mpm.add (llvm::createVerifierPass());
        // Simplify the call graph if possible (deleting unreachable blocks, etc.)
        mpm.add (llvm::createCFGSimplificationPass());
        // Change memory references to registers
        //  mpm.add (llvm::createPromoteMemoryToRegisterPass());
        mpm.add (llvm::createScalarReplAggregatesPass());
        // Combine instructions where possible -- peephole opts & bit-twiddling
        mpm.add (llvm::createInstructionCombiningPass());
        // Inline small functions
        mpm.add (llvm::createFunctionInliningPass());  // 250?
        // Eliminate early returns
        mpm.add (llvm::createUnifyFunctionExitNodesPass());
        // resassociate exprssions (a = x + (3 + y) -> a = x + y + 3)
        mpm.add (llvm::createReassociatePass());
        // Eliminate common sub-expressions
        mpm.add (llvm::createGVNPass());
        // Constant propagation with SCCP
        mpm.add (llvm::createSCCPPass());
        // More dead code elimination
        mpm.add (llvm::createAggressiveDCEPass());
        // Combine instructions where possible -- peephole opts & bit-twiddling
        mpm.add (llvm::createInstructionCombiningPass());
        // Simplify the call graph if possible (deleting unreachable blocks, etc.)
        mpm.add (llvm::createCFGSimplificationPass());
        // Try to make stuff into registers one last time.
        mpm.add (llvm::createPromoteMemoryToRegisterPass());
    }
}



void
LLVM_Util::do_optimize ()
{
#if OSL_LLVM_VERSION >= 34
    m_llvm_module_passes->run (*module());
#else
    m_llvm_module_passes->run (*module());
#endif
}



void
LLVM_Util::internalize_module_functions (const std::string &prefix,
                                         const std::vector<std::string> &exceptions,
                                         const std::vector<std::string> &moreexceptions)
{
    for (llvm::Module::iterator iter = module()->begin(); iter != module()->end(); iter++) {
        llvm::Function *sym = (llvm::Function *)(iter);
        std::string symname = sym->getName();
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
    llvm::Function *func = llvm::cast<llvm::Function>(
        module()->getOrInsertFunction (name, rettype,
                                       arg1, arg2, arg3, arg4, NULL));
    if (fastcall)
        func->setCallingConv(llvm::CallingConv::Fast);
    return func;
}



llvm::Function *
LLVM_Util::make_function (const std::string &name, bool fastcall,
                          llvm::Type *rettype,
                          const std::vector<llvm::Type*> &params,
                          bool varargs)
{
    llvm::FunctionType *functype = type_function (rettype, params, varargs);
    llvm::Constant *c = module()->getOrInsertFunction (name, functype);
    ASSERT (c && "getOrInsertFunction returned NULL");
    ASSERT_MSG (llvm::isa<llvm::Function>(c),
                "Declaration for %s is wrong, LLVM had to make a cast", name.c_str());
    llvm::Function *func = llvm::cast<llvm::Function>(c);
    if (fastcall)
        func->setCallingConv(llvm::CallingConv::Fast);
    return func;
}



llvm::Value *
LLVM_Util::current_function_arg (int a)
{
    llvm::Function::arg_iterator arg_it = current_function()->arg_begin();
    for (int i = 0;  i < a;  ++i)
        ++arg_it;
    return arg_it;
}



llvm::BasicBlock *
LLVM_Util::new_basic_block (const std::string &name)
{
    return llvm::BasicBlock::Create (context(), name, current_function());
}



llvm::BasicBlock *
LLVM_Util::push_function (llvm::BasicBlock *after)
{
    if (! after)
        after = new_basic_block ();
    m_return_block.push_back (after);
    return after;
}



void
LLVM_Util::pop_function ()
{
    ASSERT (! m_return_block.empty());
    builder().SetInsertPoint (m_return_block.back());
    m_return_block.pop_back ();
}



llvm::BasicBlock *
LLVM_Util::return_block () const
{
    ASSERT (! m_return_block.empty());
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
    ASSERT (! m_loop_step_block.empty() && ! m_loop_after_block.empty());
    m_loop_step_block.pop_back ();
    m_loop_after_block.pop_back ();
}



llvm::BasicBlock *
LLVM_Util::loop_step_block () const
{
    ASSERT (! m_loop_step_block.empty());
    return m_loop_step_block.back();
}



llvm::BasicBlock *
LLVM_Util::loop_after_block () const
{
    ASSERT (! m_loop_after_block.empty());
    return m_loop_after_block.back();
}




llvm::Type *
LLVM_Util::type_union(const std::vector<llvm::Type *> &types)
{
#if OSL_LLVM_VERSION >= 32
    llvm::DataLayout target(module());
#else
    llvm::TargetData target(module());
#endif
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
        base_type = (llvm::Type *) llvm::Type::getInt32Ty (context());
    else if (max_align == 2)
        base_type = (llvm::Type *) llvm::Type::getInt16Ty (context());
    else
        base_type = (llvm::Type *) llvm::Type::getInt8Ty (context());

    size_t array_len = union_size / target.getTypeStoreSize(base_type);
    return (llvm::Type *) llvm::ArrayType::get (base_type, array_len);
}



llvm::Type *
LLVM_Util::type_struct (const std::vector<llvm::Type *> &types,
                        const std::string &name)
{
    return llvm::StructType::create(context(), types, name);
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
LLVM_Util::constant (float f)
{
    return llvm::ConstantFP::get (context(), llvm::APFloat(f));
}



llvm::Value *
LLVM_Util::constant (int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(32,i));
}



llvm::Value *
LLVM_Util::constant (size_t i)
{
    int bits = sizeof(size_t)*8;
    return llvm::ConstantInt::get (context(), llvm::APInt(bits,i));
}



llvm::Value *
LLVM_Util::constant_bool (bool i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(1,i));
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
        std::cerr << "Bad llvm_type(" << typedesc << ")\n";
        ASSERT (0 && "not handling this type yet");
    }
    if (typedesc.arraylen)
        lt = llvm::ArrayType::get (lt, typedesc.arraylen);
    DASSERT (lt);
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
LLVM_Util::op_alloca (llvm::Type *llvmtype, int n, const std::string &name)
{
    llvm::ConstantInt* numalloc = (llvm::ConstantInt*)constant(n);
    return builder().CreateAlloca (llvmtype, numalloc, name);
}



llvm::Value *
LLVM_Util::op_alloca (const TypeDesc &type, int n, const std::string &name)
{
    return op_alloca (llvm_type(type.elementtype()), n*type.numelements(), name);
}



llvm::Value *
LLVM_Util::call_function (llvm::Value *func, llvm::Value **args, int nargs)
{
    ASSERT (func);
#if 0
    llvm::outs() << "llvm_call_function " << *func << "\n";
    llvm::outs() << nargs << " args:\n";
    for (int i = 0;  i < nargs;  ++i)
        llvm::outs() << "\t" << *(args[i]) << "\n";
#endif
    //llvm_gen_debug_printf (std::string("start ") + std::string(name));
    llvm::Value *r = builder().CreateCall (func, llvm::ArrayRef<llvm::Value *>(args, nargs));
    //llvm_gen_debug_printf (std::string(" end  ") + std::string(name));
    return r;
}



llvm::Value *
LLVM_Util::call_function (const char *name, llvm::Value **args, int nargs)
{
    llvm::Function *func = module()->getFunction (name);
    if (! func)
        std::cerr << "Couldn't find function " << name << "\n";
    return call_function (func, args, nargs);
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
    op_memset(ptr, val, constant(len), align);
}



void
LLVM_Util::op_memset (llvm::Value *ptr, int val, llvm::Value *len, int align)
{
    // memset with i32 len
    // and with an i8 pointer (dst) for LLVM-2.8
    llvm::Type* types[] = {
        (llvm::Type *) llvm::PointerType::get(llvm::Type::getInt8Ty(context()), 0),
        (llvm::Type *) llvm::Type::getInt32Ty(context())
    };

    llvm::Function* func = llvm::Intrinsic::getDeclaration (module(),
        llvm::Intrinsic::memset,
        llvm::ArrayRef<llvm::Type *>(types, sizeof(types)/sizeof(llvm::Type*)));

    // NOTE(boulos): constant(0) would return an i32
    // version of 0, but we need the i8 version. If we make an
    // ::constant(char val) though then we'll get ambiguity
    // everywhere.
    llvm::Value* fill_val = llvm::ConstantInt::get (context(),
                                                    llvm::APInt(8, val));
    // Non-volatile (allow optimizer to move it around as it wishes
    // and even remove it if it can prove it's useless)
    builder().CreateCall5 (func, ptr, fill_val, len, constant(align),
                           constant_bool(false));
}



void
LLVM_Util::op_memcpy (llvm::Value *dst, llvm::Value *src, int len, int align)
{
    // i32 len
    // and with i8 pointers (dst and src) for LLVM-2.8
    llvm::Type* types[] = {
        (llvm::Type *) llvm::PointerType::get(llvm::Type::getInt8Ty(context()), 0),
        (llvm::Type *) llvm::PointerType::get(llvm::Type::getInt8Ty(context()), 0),
        (llvm::Type *) llvm::Type::getInt32Ty(context())
    };

    llvm::Function* func = llvm::Intrinsic::getDeclaration (module(),
        llvm::Intrinsic::memcpy,
        llvm::ArrayRef<llvm::Type *>(types, sizeof(types)/sizeof(llvm::Type*)));

    // Non-volatile (allow optimizer to move it around as it wishes
    // and even remove it if it can prove it's useless)
    builder().CreateCall5 (func, dst, src,
                           constant(len), constant(align), constant_bool(false));
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
    return builder().CreateConstGEP2_32 (ptr, elem1, elem2);
}



llvm::Value *
LLVM_Util::op_add (llvm::Value *a, llvm::Value *b)
{
    if (a->getType() == type_float() && b->getType() == type_float())
        return builder().CreateFAdd (a, b);
    if (a->getType() == type_int() && b->getType() == type_int())
        return builder().CreateAdd (a, b);
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_sub (llvm::Value *a, llvm::Value *b)
{
    if (a->getType() == type_float() && b->getType() == type_float())
        return builder().CreateFSub (a, b);
    if (a->getType() == type_int() && b->getType() == type_int())
        return builder().CreateSub (a, b);
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_neg (llvm::Value *a)
{
    if (a->getType() == type_float())
        return builder().CreateFNeg (a);
    if (a->getType() == type_int())
        return builder().CreateNeg (a);
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_mul (llvm::Value *a, llvm::Value *b)
{
    if (a->getType() == type_float() && b->getType() == type_float())
        return builder().CreateFMul (a, b);
    if (a->getType() == type_int() && b->getType() == type_int())
        return builder().CreateMul (a, b);
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_float_to_int (llvm::Value* a)
{
    if (a->getType() == type_float())
        return builder().CreateFPToSI(a, type_int());
    if (a->getType() == type_int())
        return a;
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_float_to_double (llvm::Value* a)
{
    ASSERT (a->getType() == type_float());
    return builder().CreateFPExt(a, llvm::Type::getDoubleTy(context()));
}



llvm::Value *
LLVM_Util::op_int_to_float (llvm::Value* a)
{
    if (a->getType() == type_int())
        return builder().CreateSIToFP(a, type_float());
    if (a->getType() == type_float())
        return a;
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_bool_to_int (llvm::Value* a)
{
    if (a->getType() == type_bool())
        return builder().CreateZExt (a, type_int());
    if (a->getType() == type_int())
        return a;
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_make_safe_div (TypeDesc type, llvm::Value *a, llvm::Value *b)
{
    if (type.basetype == TypeDesc::FLOAT) {
        llvm::Value *div = builder().CreateFDiv (a, b);
        llvm::Value *zero = constant (0.0f);
        llvm::Value *iszero = builder().CreateFCmpOEQ (b, zero);
        return builder().CreateSelect (iszero, zero, div);
    } else {
        llvm::Value *div = builder().CreateSDiv (a, b);
        llvm::Value *zero = constant (0);
        llvm::Value *iszero = builder().CreateICmpEQ (b, zero);
        return builder().CreateSelect (iszero, zero, div);
    }
}



llvm::Value *
LLVM_Util::op_make_safe_mod (TypeDesc type, llvm::Value *a, llvm::Value *b)
{
    if (type.basetype == TypeDesc::FLOAT) {
        llvm::Value *mod = builder().CreateFRem (a, b);
        llvm::Value *zero = constant (0.0f);
        llvm::Value *iszero = builder().CreateFCmpOEQ (b, zero);
        return builder().CreateSelect (iszero, zero, mod);
    } else {
        llvm::Value *mod = builder().CreateSRem (a, b);
        llvm::Value *zero = constant (0);
        llvm::Value *iszero = builder().CreateICmpEQ (b, zero);
        return builder().CreateSelect (iszero, zero, mod);
    }
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
    if (a->getType() == type_int() && b->getType() == type_int())
        return builder().CreateAShr (a, b);  // signed int -> arithmetic shift
    ASSERT (0 && "Op has bad value type combination");
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
LLVM_Util::op_eq (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if (a->getType() == type_float())
        return ordered ? builder().CreateFCmpOEQ (a, b) : builder().CreateFCmpUEQ (a, b);
    else
        return builder().CreateICmpEQ (a, b);
}



llvm::Value *
LLVM_Util::op_ne (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if (a->getType() == type_float())
        return ordered ? builder().CreateFCmpONE (a, b) : builder().CreateFCmpUNE (a, b);
    else
        return builder().CreateICmpNE (a, b);
}



llvm::Value *
LLVM_Util::op_gt (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if (a->getType() == type_float())
        return ordered ? builder().CreateFCmpOGT (a, b) : builder().CreateFCmpUGT (a, b);
    else
        return builder().CreateICmpSGT (a, b);
}



llvm::Value *
LLVM_Util::op_lt (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if (a->getType() == type_float())
        return ordered ? builder().CreateFCmpOLT (a, b) : builder().CreateFCmpULT (a, b);
    else
        return builder().CreateICmpSLT (a, b);
}



llvm::Value *
LLVM_Util::op_ge (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if (a->getType() == type_float())
        return ordered ? builder().CreateFCmpOGE (a, b) : builder().CreateFCmpUGE (a, b);
    else
        return builder().CreateICmpSGE (a, b);
}



llvm::Value *
LLVM_Util::op_le (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if (a->getType() == type_float())
        return ordered ? builder().CreateFCmpOLE (a, b) : builder().CreateFCmpULE (a, b);
    else
        return builder().CreateICmpSLE (a, b);
}



void
LLVM_Util::write_bitcode_file (const char *filename, std::string *err)
{
    std::string local_error;
#if OSL_LLVM_VERSION >= 35
    llvm::raw_fd_ostream out (filename, err ? *err : local_error, llvm::sys::fs::F_None);
#else
    llvm::raw_fd_ostream out (filename, err ? *err : local_error);
#endif
    llvm::WriteBitcodeToFile (module(), out);
}



std::string
LLVM_Util::bitcode_string (llvm::Function *func)
{
    std::string s;
    llvm::raw_string_ostream stream (s);
    stream << (*func);
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


}; // namespace pvt
OSL_NAMESPACE_EXIT
