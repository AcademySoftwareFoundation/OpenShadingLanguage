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


#include <memory>
#include <cinttypes>
#include <OpenImageIO/thread.h>
#include <boost/thread/tss.hpp>   /* for thread_specific_ptr */

#include <OSL/oslconfig.h>
#include <OSL/llvm_util.h>

#if OSL_LLVM_VERSION < 50
#error "LLVM minimum version required for OSL is 5.0"
#endif

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/IR/LegacyPassManager.h>
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
#include <llvm/Transforms/Scalar/GVN.h>

#if OSL_LLVM_VERSION >= 70
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#endif

// additional includes for PTX generation
#include <llvm/Transforms/Utils/SymbolRewriter.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/TargetTransformInfo.h>

OSL_NAMESPACE_ENTER

namespace pvt {

typedef llvm::SectionMemoryManager LLVMMemoryManager;

typedef llvm::Error LLVMErr;


namespace {

#if OSL_LLVM_VERSION >= 60
// NOTE: This is a COPY of something internal to LLVM, but since we destroy our LLVMMemoryManager
//       via global variables we can't rely on the LLVM copy sticking around.
//       Because of this, the variable must be declared _before_ jitmm_hold so that the object stays
//       valid until after we have destroyed all our memory managers.
struct DefaultMMapper final : public llvm::SectionMemoryManager::MemoryMapper {
    llvm::sys::MemoryBlock
    allocateMappedMemory(llvm::SectionMemoryManager::AllocationPurpose Purpose,
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
#endif

static OIIO::spin_mutex llvm_global_mutex;
static bool setup_done = false;
static boost::thread_specific_ptr<LLVM_Util::PerThreadInfo> perthread_infos;
static std::vector<std::shared_ptr<LLVMMemoryManager> > jitmm_hold;
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
    LLVMMemoryManager *llvm_jitmm;
};




size_t
LLVM_Util::total_jit_memory_held ()
{
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
    
    virtual void notifyObjectLoaded(llvm::ExecutionEngine *EE, const llvm::object::ObjectFile &oi) {
        mm->notifyObjectLoaded (EE, oi);
    }

    virtual void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                                        uintptr_t RODataSize,
                                        uint32_t RODataAlign,
                                        uintptr_t RWDataSize,
                                        uint32_t RWDataAlign) {
        return mm->reserveAllocationSpace(CodeSize, CodeAlign, RODataSize, RODataAlign, RWDataSize, RWDataAlign);
    }

    virtual bool needsToReserveAllocationSpace() {
        return mm->needsToReserveAllocationSpace();
    }

    virtual void invalidateInstructionCache() {
        mm->invalidateInstructionCache();
    }

    // Common
    virtual ~MemoryManager() {}

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
    virtual void deregisterEHFrames() {
        mm->deregisterEHFrames();
    }

    virtual uint64_t getSymbolAddress(const std::string &Name) {
        return mm->getSymbolAddress (Name);
    }
    virtual bool finalizeMemory(std::string *ErrMsg = 0) {
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



LLVM_Util::LLVM_Util (int debuglevel)
    : m_debug(debuglevel), m_thread(NULL),
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
#if OSL_LLVM_VERSION >= 60
            m_thread->llvm_jitmm = new LLVMMemoryManager(&llvm_default_mapper);
#else
            m_thread->llvm_jitmm = new LLVMMemoryManager();
#endif
            ASSERT (m_thread->llvm_jitmm);
            jitmm_hold.emplace_back (m_thread->llvm_jitmm);
        }
        // Hold the REAL manager and use it as an argument later
        m_llvm_jitmm = m_thread->llvm_jitmm;
    }

    m_llvm_context = m_thread->llvm_context;

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
}


/// Return the current IR builder, create a new one (for the current
/// function) if necessary.
LLVM_Util::IRBuilder &
LLVM_Util::builder () {
    if (! m_builder)
        new_builder ();
    return *m_builder;
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
    llvm::EngineBuilder engine_builder ((std::unique_ptr<llvm::Module>(module())));

    engine_builder.setEngineKind (llvm::EngineKind::JIT);
    engine_builder.setErrorStr (err);

    // We are actually holding a LLVMMemoryManager
    engine_builder.setMCJITMemoryManager (std::unique_ptr<llvm::RTDyldMemoryManager>
        (new MemoryManager(m_llvm_jitmm)));

    engine_builder.setOptLevel (llvm::CodeGenOpt::Default);

    m_llvm_exec = engine_builder.create();
    if (! m_llvm_exec)
        return NULL;

    // These magic lines will make it so that enough symbol information
    // is injected so that running vtune will kinda tell you which shaders
    // you're in, and sometimes which function (only for functions that don't
    // get inlined. There doesn't seem to be any perf hit from this, either
    // in code quality or JIT time. It is only enabled, however, if your copy
    // of LLVM was build with -DLLVM_USE_INTEL_JITEVENTS=ON, otherwise
    // createIntelJITEventListener() is a stub that just returns nullptr.
    auto vtuneProfiler = llvm::JITEventListener::createIntelJITEventListener();
    if (vtuneProfiler)
        m_llvm_exec->RegisterJITEventListener (vtuneProfiler);

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
    DASSERT (func && "passed NULL to getPointerToFunction");
    llvm::ExecutionEngine *exec = execengine();
    exec->finalizeObject ();
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

    // Construct the per-function passes and module-wide (interprocedural
    // optimization) passes.

    m_llvm_func_passes = new llvm::legacy::FunctionPassManager(module());
    llvm::legacy::FunctionPassManager &fpm = (*m_llvm_func_passes);

    m_llvm_module_passes = new llvm::legacy::PassManager;
    llvm::legacy::PassManager &mpm = (*m_llvm_module_passes);

    if (optlevel >= 1 && optlevel <= 3) {
        // For LLVM 3.0 and higher, llvm_optimize 1-3 means to use the
        // same set of optimizations as clang -O1, -O2, -O3
        llvm::PassManagerBuilder builder;
        builder.OptLevel = optlevel;
        builder.Inliner = llvm::createFunctionInliningPass();
        // builder.DisableUnrollLoops = true;
        builder.populateFunctionPassManager (fpm);
        builder.populateModulePassManager (mpm);
    } else {
        // Unknown choices for llvm_optimize: use the same basic
        // set of passes that we always have.

        // Always add verifier?
        mpm.add (llvm::createVerifierPass());
        // Simplify the call graph if possible (deleting unreachable blocks, etc.)
        mpm.add (llvm::createCFGSimplificationPass());
        // Change memory references to registers
        //  mpm.add (llvm::createPromoteMemoryToRegisterPass());

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
LLVM_Util::do_optimize (std::string *out_err)
{
    ASSERT(m_llvm_module && "No module to optimize!");

#if !defined(OSL_FORCE_BITCODE_PARSE)
    LLVMErr err = m_llvm_module->materializeAll();
    if (error_string(std::move(err), out_err))
        return;
#endif

    m_llvm_func_passes->doInitialization();
    m_llvm_module_passes->run (*m_llvm_module);
    m_llvm_func_passes->doFinalization();
}



void
LLVM_Util::internalize_module_functions (const std::string &prefix,
                                         const std::vector<std::string> &exceptions,
                                         const std::vector<std::string> &moreexceptions)
{
    for (llvm::Function& func : module()->getFunctionList()) {
        llvm::Function *sym = &func;
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
    llvm::Constant *c = module()->getOrInsertFunction (name, functype);
    ASSERT (c && "getOrInsertFunction returned NULL");
    ASSERT_MSG (llvm::isa<llvm::Function>(c),
                "Declaration for %s is wrong, LLVM had to make a cast", name.c_str());
    llvm::Function *func = llvm::cast<llvm::Function>(c);
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
    builder().CreateMemSet (ptr, builder().getInt8((unsigned char)val),
                            uint64_t(len), (unsigned)align);
}



void
LLVM_Util::op_memset (llvm::Value *ptr, int val, llvm::Value *len, int align)
{
    builder().CreateMemSet (ptr, builder().getInt8((unsigned char)val),
                            len, (unsigned)align);
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
#if OSL_LLVM_VERSION >= 70
    builder().CreateMemCpy (dst, (unsigned)dstalign, src, (unsigned)srcalign,
                            uint64_t(len));
#else
    builder().CreateMemCpy (dst, src, uint64_t(len),
                            std::min ((unsigned)dstalign, (unsigned)srcalign));
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
LLVM_Util::op_div (llvm::Value *a, llvm::Value *b)
{
    if (a->getType() == type_float() && b->getType() == type_float())
        return builder().CreateFDiv (a, b);
    if (a->getType() == type_int() && b->getType() == type_int())
        return builder().CreateSDiv (a, b);
    ASSERT (0 && "Op has bad value type combination");
}



llvm::Value *
LLVM_Util::op_mod (llvm::Value *a, llvm::Value *b)
{
    if (a->getType() == type_float() && b->getType() == type_float())
        return builder().CreateFRem (a, b);
    if (a->getType() == type_int() && b->getType() == type_int())
        return builder().CreateSRem (a, b);
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
LLVM_Util::op_int_to_longlong (llvm::Value* a)
{
    ASSERT (a->getType() == type_int());
    return builder().CreateSExt(a, llvm::Type::getInt64Ty(context()));
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
    std::error_code local_error;
    llvm::raw_fd_ostream out (filename, local_error, llvm::sys::fs::F_None);
    if (! out.has_error()) {
#if OSL_LLVM_VERSION >= 70
        llvm::WriteBitcodeToFile (*module(), out);
#else
        llvm::WriteBitcodeToFile (module(), out);
#endif
        if (err && local_error)
            *err = local_error.message ();
    }
}



bool
LLVM_Util::ptx_compile_group (llvm::Module* lib_module, const std::string& name,
                              std::string& out)
{
    std::string target_triple = module()->getTargetTriple();
    ASSERT (lib_module->getTargetTriple() == target_triple &&
            "PTX compile error: Shader and renderer bitcode library targets do not match");

    // Create a new empty module to hold the linked shadeops and compiled
    // ShaderGroup
    llvm::Module* linked_module = new_module (name.c_str());

    // First, link in the cloned ShaderGroup module
#if OSL_LLVM_VERSION >= 70
    std::unique_ptr<llvm::Module> mod_ptr = llvm::CloneModule (*module());
#else
    std::unique_ptr<llvm::Module> mod_ptr = llvm::CloneModule (module());
#endif
    bool failed = llvm::Linker::linkModules (*linked_module, std::move (mod_ptr));
    if (failed) {
        ASSERT (0 && "PTX compile error: Unable to link group module");
    }

    // Second, link in the shadeops library, keeping only the functions that are needed
    std::unique_ptr<llvm::Module> lib_ptr (lib_module);
    failed = llvm::Linker::linkModules (*linked_module, std::move (lib_ptr));

    // Internalize the Globals to match code generated by NVCC
    for (auto& g_var : linked_module->globals()) {
        g_var.setLinkage(llvm::GlobalValue::InternalLinkage);
    }

    // Verify that the NVPTX target has been initialized
    std::string error;
    const llvm::Target* llvm_target =
        llvm::TargetRegistry::lookupTarget (target_triple, error);
    if(! llvm_target) {
        ASSERT (0 && "PTX compile error: LLVM Target is not initialized");
    }

    llvm::TargetOptions  options;
    options.AllowFPOpFusion                        = llvm::FPOpFusion::Standard;
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
    if (! target_machine) {
        ASSERT(0 && "PTX compile error: Unable to create target machine -- is NVPTX enabled in LLVM?");
    }

    // Setup the optimzation passes
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
#if OSL_LLVM_VERSION >= 70
    target_machine->addPassesToEmitFile (mod_pm, assembly_stream,
                                         nullptr,  // FIXME: Correct?
                                         llvm::TargetMachine::CGFT_AssemblyFile);
#else
    target_machine->addPassesToEmitFile (mod_pm, assembly_stream,
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


}; // namespace pvt
OSL_NAMESPACE_EXIT
