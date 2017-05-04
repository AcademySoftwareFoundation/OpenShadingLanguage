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
#include <OpenImageIO/thread.h>
#include <boost/thread/tss.hpp>   /* for thread_specific_ptr */
#include <unordered_map>

#include "OSL/oslconfig.h"
#include "OSL/llvm_util.h"
#include "OSL/wide.h"

#if OSL_LLVM_VERSION < 34
#error "LLVM minimum version required for OSL is 3.4"
#endif

#if OSL_LLVM_VERSION >= 35 && OSL_CPLUSPLUS_VERSION < 11
#error "LLVM >= 3.5 requires C++11 or newer"
#endif

// Use MCJIT for LLVM 3.6 and beyind, old JIT for earlier
#define USE_MCJIT   (OSL_LLVM_VERSION >= 36)
#define USE_OLD_JIT (OSL_LLVM_VERSION <  36)

#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DataLayout.h>
#if OSL_LLVM_VERSION >= 35
#  include <llvm/Linker/Linker.h>
#  include <llvm/Support/FileSystem.h>
#else
#  include <llvm/Linker.h>
#endif
#include <llvm/Support/ErrorOr.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetRegistry.h>

#if OSL_LLVM_VERSION < 40
#  include <llvm/Bitcode/ReaderWriter.h>
#else
#  include <llvm/Bitcode/BitcodeReader.h>
#  include <llvm/Bitcode/BitcodeWriter.h>
#endif
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#if USE_MCJIT
#  include <llvm/ExecutionEngine/MCJIT.h>
#endif
#if USE_OLD_JIT
#  include <llvm/ExecutionEngine/JIT.h>
#  include <llvm/ExecutionEngine/JITMemoryManager.h>
#endif
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/PrettyStackTrace.h>
#if OSL_LLVM_VERSION >= 35
#  include <llvm/IR/Verifier.h>
#else
#  include <llvm/Analysis/Verifier.h>
#endif
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#if OSL_LLVM_VERSION >= 36
#  include <llvm/ExecutionEngine/SectionMemoryManager.h>
#endif
#if OSL_LLVM_VERSION >= 39
#  include <llvm/Transforms/Scalar/GVN.h>
#endif

OSL_NAMESPACE_ENTER

namespace pvt {

#if USE_OLD_JIT
    typedef llvm::JITMemoryManager LLVMMemoryManager;
#else
    typedef llvm::SectionMemoryManager LLVMMemoryManager;
#endif

#if OSL_LLVM_VERSION >= 35
#if OSL_LLVM_VERSION < 40
    typedef std::error_code LLVMErr;
#else
    typedef llvm::Error LLVMErr;
#endif
#endif


namespace {
static OIIO::spin_mutex llvm_global_mutex;
static bool setup_done = false;
static boost::thread_specific_ptr<LLVM_Util::PerThreadInfo> perthread_infos;
static std::vector<std::shared_ptr<LLVMMemoryManager> > jitmm_hold;

static struct DebugInfo {
	llvm::DICompileUnit *TheCU;
	llvm::DIType *DblTy;
	std::vector<llvm::DIScope *> LexicalBlocks;

	typedef std::unordered_map<std::string, llvm::DISubprogram *> ScopeByNameType;
	ScopeByNameType ScopeByName;
	
	typedef std::unordered_map<std::string, llvm::DIFile *> FileByNameType;
	FileByNameType FileByName;
	
	
	//void emitLocation(ExprAST *AST);
	//llvm::DIType *getDoubleTy();
} TheDebugInfo;

llvm::DIFile * getFileFor(llvm::DIBuilder* diBuilder, const std::string &file_name) {
	auto iter = TheDebugInfo.FileByName.find(file_name);
	if(iter == TheDebugInfo.FileByName.end()) {
		//std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>CREATING FILE<<<<<<<<<<<<<<<<<<<<<<<<< " << file_name << std::endl;
		llvm::DIFile *file = diBuilder->createFile(
				//TheDebugInfo.TheCU->getFilename(), TheDebugInfo.TheCU->getDirectory());
				file_name, ".\\");
		//llvm::DIScope *FContext = Unit;
		TheDebugInfo.FileByName.insert(std::make_pair(file_name,file));
		return file;
	}
	return iter->second;
}

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
#if USE_OLD_JIT
    for (size_t i = 0;  i < jitmm_hold.size();  ++i) {
        LLVMMemoryManager *mm = jitmm_hold[i].get();
        if (mm)
            jitmem += mm->GetDefaultCodeSlabSize() * mm->GetNumCodeSlabs()
                    + mm->GetDefaultDataSlabSize() * mm->GetNumDataSlabs()
                    + mm->GetDefaultStubSlabSize() * mm->GetNumStubSlabs();
    }
#endif
    return jitmem;
}



/// MemoryManager - Create a shell that passes on requests
/// to a real LLVMMemoryManager underneath, but can be retained after the
/// dummy is destroyed.  Also, we don't pass along any deallocations.
class LLVM_Util::MemoryManager : public LLVMMemoryManager {
protected:
    LLVMMemoryManager *mm;  // the real one
public:

#if USE_OLD_JIT // llvm::JITMemoryManager
    MemoryManager(LLVMMemoryManager *realmm) : mm(realmm) { HasGOT = realmm->isManagingGOT(); }

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

    virtual void notifyObjectLoaded(llvm::ExecutionEngine *EE, const llvm::ObjectImage *oi) {
        mm->notifyObjectLoaded (EE, oi);
    }

#else // MCJITMemoryManager

    MemoryManager(LLVMMemoryManager *realmm) : mm(realmm) {}
    
    virtual void notifyObjectLoaded(llvm::ExecutionEngine *EE, const llvm::object::ObjectFile &oi) {
        mm->notifyObjectLoaded (EE, oi);
    }

#if OSL_LLVM_VERSION <= 37
  virtual void reserveAllocationSpace(
    uintptr_t CodeSize, uintptr_t DataSizeRO, uintptr_t DataSizeRW) {
        return mm->reserveAllocationSpace(CodeSize, DataSizeRO, DataSizeRW);
}
#else
    virtual void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                                        uintptr_t RODataSize,
                                        uint32_t RODataAlign,
                                        uintptr_t RWDataSize,
                                        uint32_t RWDataAlign) {
        return mm->reserveAllocationSpace(CodeSize, CodeAlign, RODataSize, RODataAlign, RWDataSize, RWDataAlign);
    }
#endif

    virtual bool needsToReserveAllocationSpace() {
        return mm->needsToReserveAllocationSpace();
    }

    virtual void invalidateInstructionCache() {
        mm->invalidateInstructionCache();
    }
    
#endif

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
    virtual void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) {
        mm->deregisterEHFrames(Addr, LoadAddr, Size);
    }
    virtual uint64_t getSymbolAddress(const std::string &Name) {
        return mm->getSymbolAddress (Name);
    }
    virtual bool finalizeMemory(std::string *ErrMsg = 0) {
        return mm->finalizeMemory (ErrMsg);
    }
};



#if OSL_LLVM_VERSION <= 38
class LLVM_Util::IRBuilder : public llvm::IRBuilder<true,llvm::ConstantFolder,
                                        llvm::IRBuilderDefaultInserter<true> > {
    typedef llvm::IRBuilder<true, llvm::ConstantFolder,
                                  llvm::IRBuilderDefaultInserter<true> > Base;
public:
    IRBuilder(llvm::BasicBlock *TheBB) : Base(TheBB) {}
};
#else
class LLVM_Util::IRBuilder : public llvm::IRBuilder<llvm::ConstantFolder,
                                               llvm::IRBuilderDefaultInserter> {
    typedef llvm::IRBuilder<llvm::ConstantFolder,
                            llvm::IRBuilderDefaultInserter> Base;
public:
    IRBuilder(llvm::BasicBlock *TheBB) : Base(TheBB) {}
};
#endif



LLVM_Util::LLVM_Util (int debuglevel)
    : m_debug(debuglevel), m_thread(NULL),
      m_llvm_context(NULL), m_llvm_module(NULL),
      m_llvm_debug_builder(NULL),
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
#if USE_OLD_JIT
            m_thread->llvm_jitmm = llvm::JITMemoryManager::CreateDefaultMemManager();
#else
            m_thread->llvm_jitmm = new LLVMMemoryManager;
#endif
            ASSERT (m_thread->llvm_jitmm);
            jitmm_hold.push_back (std::shared_ptr<LLVMMemoryManager>(m_thread->llvm_jitmm));
        }
#if USE_OLD_JIT
        m_llvm_jitmm = new MemoryManager(m_thread->llvm_jitmm);
#else
        // Hold the REAL manager and use it as an argument later
        m_llvm_jitmm = reinterpret_cast<MemoryManager*>(m_thread->llvm_jitmm);
#endif
    }

    m_llvm_context = m_thread->llvm_context;

    // Set up aliases for types we use over and over
    m_llvm_type_float = (llvm::Type *) llvm::Type::getFloatTy (*m_llvm_context);
    m_llvm_type_double = (llvm::Type *) llvm::Type::getDoubleTy (*m_llvm_context);
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

    // Setup up wide aliases
    m_vector_width = SimdLaneCount;
    // TODO:  why are there casts to the base class llvm::Type *?  
    m_llvm_type_wide_float = llvm::VectorType::get(m_llvm_type_float, m_vector_width);
    m_llvm_type_wide_double = llvm::VectorType::get(m_llvm_type_double, m_vector_width);
    m_llvm_type_wide_int = llvm::VectorType::get(m_llvm_type_int, m_vector_width);
    m_llvm_type_wide_bool = llvm::VectorType::get(m_llvm_type_bool, m_vector_width);
    m_llvm_type_wide_char = llvm::VectorType::get(m_llvm_type_char, m_vector_width);
    
    m_llvm_type_wide_char_ptr = llvm::PointerType::get(m_llvm_type_wide_char, 0);    
    m_llvm_type_wide_void_ptr = llvm::VectorType::get(m_llvm_type_void_ptr, m_vector_width);
    m_llvm_type_wide_int_ptr = llvm::PointerType::get(m_llvm_type_wide_int, 0);
    m_llvm_type_wide_bool_ptr = llvm::PointerType::get(m_llvm_type_wide_bool, 0);

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

#if OSL_LLVM_VERSION < 35
    // enable it to be thread-safe
    llvm::llvm_start_multithreaded ();
#endif
// new versions (>=3.5)don't need this anymore


#if USE_MCJIT
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeDisassembler();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
    LLVMLinkInMCJIT();
#else
    llvm::InitializeNativeTarget();
#endif

    if (debug()) {
#if OSL_LLVM_VERSION <= 36
# define OSL_TGT_DEF(t) t->
        for (llvm::TargetRegistry::iterator t = llvm::TargetRegistry::begin();
             t != llvm::TargetRegistry::end();  ++t)
#else
# define OSL_TGT_DEF(t) t.
        for (auto t : llvm::TargetRegistry::targets())
#endif
        {
            std::cout << "Target: '" << OSL_TGT_DEF(t)getName() << "' "
                      << OSL_TGT_DEF(t)getShortDescription() << "\n";
        }
#undef OSL_TGT_DEF
        std::cout << "\n";
    }

    setup_done = true;
}



llvm::Module *
LLVM_Util::new_module (const char *id)
{
    return new llvm::Module(id, context());
}

void 
LLVM_Util::enable_debug_info() {
	module()->addModuleFlag(llvm::Module::Error, "Debug Info Version",
			llvm::DEBUG_METADATA_VERSION);

	unsigned int modulesDebugInfoVersion = 0;
	if (auto *Val = llvm::mdconst::dyn_extract_or_null < llvm::ConstantInt
			> (module()->getModuleFlag("Debug Info Version"))) {
		modulesDebugInfoVersion = Val->getZExtValue();
	}

//	std::cout
//			<< "------------------>enable_debug_info<-----------------------------module flag['Debug Info Version']= "
//			<< modulesDebugInfoVersion << std::endl;
}

void 
LLVM_Util::set_debug_info(const std::string &function_name) {

	m_llvm_debug_builder = (new llvm::DIBuilder(*m_llvm_module));

	TheDebugInfo.TheCU = m_llvm_debug_builder->createCompileUnit(
			llvm::dwarf::DW_LANG_C, 
# if OSL_LLVM_VERSION >= 40
			m_llvm_debug_builder->createFile("JIT", // filename
					"." // directory
					),
#else			
			"JIT", // filename
			".", // directory
#endif
			"OSLv1.9", // Identify the producer of debugging information and code. Usually this is a compiler version string.
			0, // Identify the producer of debugging information and code. Usually this is a compiler version string.
			"", // This string lists command line options. This string is directly embedded in debug info output which may be used by a tool analyzing generated debugging information.
			1900); // This indicates runtime version for languages like Objective-C
	
	llvm::DIFile * file = getFileFor(m_llvm_debug_builder, function_name); 
	
			unsigned int method_line = 0;
				unsigned int method_scope_line = 0;
				
				
				static llvm::DISubroutineType *subType;
				{
					llvm::SmallVector<llvm::Metadata *, 8> EltTys;
					//llvm::DIType *DblTy = KSTheDebugInfo.getDoubleTy();
					llvm::DIType *debug_double_type = m_llvm_debug_builder->createBasicType(
# if OSL_LLVM_VERSION >= 40
							"double", 64, llvm::dwarf::DW_ATE_float);
#else
					"double", 64, 64, llvm::dwarf::DW_ATE_float);
#endif
			#if 0
					// Add the result type.
					EltTys.push_back(DblTy);

					for (unsigned i = 0, e = NumArgs; i != e; ++i)
					EltTys.push_back(DblTy);
			#endif
					EltTys.push_back(debug_double_type);
					EltTys.push_back(debug_double_type);

					subType = m_llvm_debug_builder->createSubroutineType(
							m_llvm_debug_builder->getOrCreateTypeArray(EltTys));
				}

				llvm::DISubprogram *function = m_llvm_debug_builder->createFunction(file,
						function_name, llvm::StringRef(), file, method_line, subType,
						false /*isLocalToUnit*/, true /*bool isDefinition*/, method_scope_line,
						llvm::DINode::FlagPrototyped, false);		
				
		current_function()->setSubprogram(function);							
	
	
}

void 
LLVM_Util::set_debug_location(const std::string &source_file_name, const std::string & method_name, int sourceline)
{
		
	llvm::DISubprogram *sp = current_function()->getSubprogram();
	ASSERT(sp != NULL);
	
	
	const llvm::DebugLoc & current_debug_location = m_builder->getCurrentDebugLocation();
	bool newDebugLocation = true;
	if (current_debug_location)
	{
		if(sourceline == current_debug_location.getLine()) {		
			newDebugLocation = false;
		}
	} 
	
	if (newDebugLocation)
	{
		//std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>newDebugLocation<<<<<<<<<<<<<<<<<<<<<<<<< " << sourceline << std::endl;
		llvm::DebugLoc debug_location =
				llvm::DebugLoc::get(static_cast<unsigned int>(sourceline),
						static_cast<unsigned int>(0), /* column? */
						sp);
		m_builder->SetCurrentDebugLocation(debug_location);
	}
}

void 
LLVM_Util::clear_debug_info() {
	std::cout << "LLVM_Util::clear_debug_info" << std::endl;
	m_builder->SetCurrentDebugLocation(llvm::DebugLoc());
	m_llvm_debug_builder->finalize();
}




#if OSL_LLVM_VERSION >= 35
#if OSL_LLVM_VERSION < 40
inline bool error_string (const std::error_code &err, std::string *str) {
    if (err) {
        if (str) *str = err.message();
        return true;
    }
    return false;
}
#else
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
#endif
#endif



llvm::Module *
LLVM_Util::module_from_bitcode (const char *bitcode, size_t size,
                                const std::string &name, std::string *err)
{
    if (err)
        err->clear();

#if OSL_LLVM_VERSION <= 35 /* Old JIT vvvvvvvvvvvvvvvvvvvvvvvvvvvv */
    llvm::MemoryBuffer* buf =
        llvm::MemoryBuffer::getMemBuffer (llvm::StringRef(bitcode, size), name);

    // Create a lazily deserialized IR module
    // This can only be done for old JIT
# if OSL_LLVM_VERSION >= 35
    llvm::Module *m = llvm::getLazyBitcodeModule (buf, context()).get();
# else
    llvm::Module *m = llvm::getLazyBitcodeModule (buf, context(), err);
# endif
    // don't delete buf, the module has taken ownership of it

#if 0
    // Debugging: print all functions in the module
    for (llvm::Module::iterator i = m->begin(); i != m->end(); ++i)
        std::cout << "  found " << i->getName().data() << "\n";
#endif
    return m;
#endif /* End of LLVM <= 3.5 Old JIT section ^^^^^^^^^^^^^^^^^^^^^ */


#if OSL_LLVM_VERSION >= 36  /* MCJIT vvvvvvvvvvvvvvvvvvvvvvvvvvvvv */
# if OSL_LLVM_VERSION >= 40
    typedef llvm::Expected<std::unique_ptr<llvm::Module> > ErrorOrModule;
# else
    typedef llvm::ErrorOr<std::unique_ptr<llvm::Module> > ErrorOrModule;
# endif

# if OSL_LLVM_VERSION >= 40 || defined(OSL_FORCE_BITCODE_PARSE)
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

# else /* !OSL_FORCE_BITCODE_PARSE */
    std::unique_ptr<llvm::MemoryBuffer> buf (
        llvm::MemoryBuffer::getMemBuffer (llvm::StringRef(bitcode, size), name, false));
    ErrorOrModule ModuleOrErr = llvm::getLazyBitcodeModule(std::move(buf), context());
# endif

    if (err) {
# if OSL_LLVM_VERSION >= 40
        error_string(ModuleOrErr.takeError(), err);
# else
        error_string(ModuleOrErr.getError(), err);
# endif
    }
    llvm::Module *m = ModuleOrErr ? ModuleOrErr->release() : nullptr;
# if 0
    // Debugging: print all functions in the module
    for (llvm::Module::iterator i = m->begin(); i != m->end(); ++i)
        std::cout << "  found " << i->getName().data() << "\n";
# endif
    
    return m;

#endif /* MCJIT ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
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
# if OSL_LLVM_VERSION >= 36
    llvm::EngineBuilder engine_builder ((std::unique_ptr<llvm::Module>(module())));
# else /* < 36: */
    llvm::EngineBuilder engine_builder (module());
# endif

    engine_builder.setEngineKind (llvm::EngineKind::JIT);
    engine_builder.setErrorStr (err);
    //engine_builder.setRelocationModel(llvm::Reloc::PIC_);
    //engine_builder.setCodeModel(llvm::CodeModel::Default);
    engine_builder.setVerifyModules(true);

#if USE_OLD_JIT
    engine_builder.setJITMemoryManager (m_llvm_jitmm);
    // N.B. createJIT will take ownership of the the JITMemoryManager!
    engine_builder.setUseMCJIT (0);
#else
    // We are actually holding a LLVMMemoryManager
    engine_builder.setMCJITMemoryManager (std::unique_ptr<llvm::RTDyldMemoryManager>
        (new MemoryManager(reinterpret_cast<LLVMMemoryManager*>(m_llvm_jitmm))));
#endif /* USE_OLD_JIT */

    
    //engine_builder.setOptLevel (llvm::CodeGenOpt::Default);
    engine_builder.setOptLevel (llvm::CodeGenOpt::Aggressive);
    
#if 1
    llvm::TargetOptions options;
    options.LessPreciseFPMADOption = true;
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    options.UnsafeFPMath = true;

    #if OSL_LLVM_VERSION < 40
    // Turn off approximate reciprocals for division. It's too
    // inaccurate even for us. In LLVM 4.0+ this moved to be a
    // function attribute.
    options.Reciprocals.setDefaults("all", false, 0);
    #endif

    options.NoInfsFPMath = true;
    options.NoNaNsFPMath = true;
    options.HonorSignDependentRoundingFPMathOption = false;
    options.NoZerosInBSS = false;
    options.GuaranteedTailCallOpt = false;
    options.StackAlignmentOverride = 0;
    options.FunctionSections = true;
    options.UseInitArray = false;
    bool use_soft_float_abi = false;
    options.FloatABIType =
        use_soft_float_abi ? llvm::FloatABI::Soft : llvm::FloatABI::Hard;
    #if LLVM_VERSION >= 39
    // Not supported by older linkers
    options.RelaxELFRelocations = false;    
    #endif    
    
    //options.PrintMachineCode = true;
    engine_builder.setTargetOptions(options);
    
    
#endif
    
#if 0
//    llvm::TargetOptions options = InitTargetOptionsFromCodeGenFlags();
    llvm::TargetOptions options;
    options.LessPreciseFPMADOption = EnableFPMAD;
   options.AllowFPOpFusion = FuseFPOps;
   options.Reciprocals = TargetRecip(ReciprocalOps);
   options.UnsafeFPMath = EnableUnsafeFPMath;
   options.NoInfsFPMath = EnableNoInfsFPMath;
   options.NoNaNsFPMath = EnableNoNaNsFPMath;
   options.HonorSignDependentRoundingFPMathOption =
		   EnableHonorSignDependentRoundingFPMath;
   if (FloatABIForCalls != FloatABI::Default)
      options.FloatABIType = FloatABIForCalls;
   options.NoZerosInBSS = DontPlaceZerosInBSS;
   options.GuaranteedTailCallOpt = EnableGuaranteedTailCallOpt;
   //options.StackAlignmentOverride = OverrideStackAlignment;
   options.StackAlignmentOverride = 32;
   //options.PositionIndependentExecutable = EnablePIE;
   options.UseInitArray = !UseCtors;
   options.DataSections = DataSections;
   options.FunctionSections = FunctionSections;
   options.UniqueSectionNames = UniqueSectionNames;
   options.EmulatedTLS = EmulatedTLS;
   
   options.MCOptions = InitMCTargetOptionsFromFlags();
   options.JTType = JTableType;
   
   options.ThreadModel = llvm::ThreadModel::Single;
   options.EABIVersion = EABIVersion;
   //options.EABIVersion = EABI::EABI4;
   options.DebuggerTuning = DebuggerTuningOpt;   
   options.RelaxELFRelocations = false;
   //options.PrintMachineCode = true;
   engine_builder.setTargetOptions(options);
#endif
    
   
   enum TargetISA
   {
	   TargetISA_UNLIMITTED,
	   TargetISA_SSE4_2,
	   TargetISA_AVX,
	   TargetISA_AVX2,
	   TargetISA_AVX512
   };
   
   TargetISA oslIsa = TargetISA_UNLIMITTED;
   const char * oslIsaString = std::getenv("OSL_ISA");
   if (oslIsaString != NULL) {
	   if (strcmp(oslIsaString, "SSE4.2") == 0)
	   {
		   oslIsa = TargetISA_SSE4_2;
	   } else if (strcmp(oslIsaString, "AVX") == 0)
	   {
		   oslIsa = TargetISA_AVX;
	   } else if (strcmp(oslIsaString, "AVX2") == 0)
	   {
		   oslIsa = TargetISA_AVX2;
	   } else if (strcmp(oslIsaString, "AVX512") == 0)
	   {
		   oslIsa = TargetISA_AVX512;
	   }
   }
   
    //engine_builder.setMArch("core-avx2");
    std::cout << std::endl<< "llvm::sys::getHostCPUName()>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << llvm::sys::getHostCPUName().str() << std::endl;
    //engine_builder.setMCPU(llvm::sys::getHostCPUName());
    //engine_builder.setMCPU("skylake-avx512");
    //engine_builder.setMCPU("broadwell");
    engine_builder.setMArch("x86-64");    

//    bool disableFMA = true;
    bool disableFMA = false;
    const char * oslNoFmaString = std::getenv("OSL_NO_FMA");
    if (oslNoFmaString != NULL) {
 	   if ((strcmp(oslNoFmaString, "1") == 0) || 
		   (strcmp(oslNoFmaString, "y") == 0) ||
		   (strcmp(oslNoFmaString, "Y") == 0) ||
		   (strcmp(oslNoFmaString, "yes") == 0) ||
		   (strcmp(oslNoFmaString, "t") == 0) ||
		   (strcmp(oslNoFmaString, "true") == 0) ||
		   (strcmp(oslNoFmaString, "T") == 0) ||
		   (strcmp(oslNoFmaString, "TRUE") == 0))
 	   {
 		  disableFMA = true;
 	   } 
    }
    
    llvm::StringMap< bool > cpuFeatures;
    if (llvm::sys::getHostCPUFeatures(cpuFeatures)) {
		std::cout << std::endl<< "llvm::sys::getHostCPUFeatures()>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
		std::vector<std::string> attrvec;
		for (auto &cpuFeature : cpuFeatures) 
		{
			//auto enabled = (cpuFeature.second && (cpuFeature.first().str().find("512") == std::string::npos)) ? "+" : "-";
			auto enabled = (cpuFeature.second) ? "+" : "-";
			//std::cout << cpuFeature.first().str()  << " is " << enabled << std::endl;
			
			if (oslIsa == TargetISA_UNLIMITTED) {				
				if (!disableFMA || std::string("fma") != cpuFeature.first().str()) {
					attrvec.push_back(enabled + cpuFeature.first().str());
				}
			}
		}
		//The particular format of the names are target dependent, and suitable for passing as -mattr to the target which matches the host.
	//    const char *mattr[] = {"avx"};
	//    std::vector<std::string> attrvec (mattr, mattr+1);

		m_supports_masked_stores = false;
		
		switch(oslIsa) {
		case TargetISA_SSE4_2:
			attrvec.push_back("+sse4.2");
			std::cout << "Intended OSL ISA: SSE4.2" << std::endl;
			break;
		case TargetISA_AVX:
			m_supports_masked_stores = true;
			attrvec.push_back("+avx");
			std::cout << "Intended OSL ISA: AVX" << std::endl;
			break;		
		case TargetISA_AVX2:
			m_supports_masked_stores = true;
			attrvec.push_back("+avx2");
			std::cout << "Intended OSL ISA: AVX2" << std::endl;
			break;		
		case TargetISA_AVX512:
			m_supports_masked_stores = true;
			attrvec.push_back("+avx512f");
			attrvec.push_back("+avx512dq");
			attrvec.push_back("+avx512bw");
			attrvec.push_back("+avx512vl");
			attrvec.push_back("+avx512cd");
			attrvec.push_back("+avx512f");
			
			std::cout << "Intended OSL ISA: AVX512" << std::endl;
			break;		
		case TargetISA_UNLIMITTED:		
		default:
			break;
		};
		
	    if (disableFMA) {
			attrvec.push_back("-fma");
	    }
		engine_builder.setMAttrs(attrvec);
		
    }
    

    m_llvm_exec = engine_builder.create();        
    if (! m_llvm_exec)
        return NULL;
    
    const llvm::DataLayout & data_layout = m_llvm_exec->getDataLayout();
    //std::cout << "data_layout.getStringRepresentation()=" << data_layout.getStringRepresentation() << std::endl;
    		
    
    TargetMachine * target_machine = m_llvm_exec->getTargetMachine();
    //std::cout << "target_machine.getTargetCPU()=" << target_machine->getTargetCPU().str() << std::endl;
	std::cout << "target_machine.getTargetFeatureString ()=" << target_machine->getTargetFeatureString ().str() << std::endl;
	//std::cout << "target_machine.getTargetTriple ()=" << target_machine->getTargetTriple().str() << std::endl;
    

    llvm::JITEventListener* vtuneProfiler = llvm::JITEventListener::createIntelJITEventListener();
    assert (vtuneProfiler != NULL);
    m_llvm_exec->RegisterJITEventListener(vtuneProfiler);
      
    // Force it to JIT as soon as we ask it for the code pointer,
    // don't take any chances that it might JIT lazily, since we
    // will be stealing the JIT code memory from under its nose and
    // destroying the Module & ExecutionEngine.
    m_llvm_exec->DisableLazyCompilation ();
    return m_llvm_exec;
}


void
LLVM_Util::dump_struct_data_layout(llvm::Type *Ty)
{
	ASSERT(Ty);
	ASSERT(Ty->isStructTy());
			
	llvm::StructType *structTy = static_cast<llvm::StructType *>(Ty);
    const llvm::DataLayout & data_layout = m_llvm_exec->getDataLayout();
    
    int number_of_elements = structTy->getNumElements();


	const StructLayout * layout = data_layout.getStructLayout (structTy);
	std::cout << "dump_struct_data_layout: getSizeInBytes(" << layout->getSizeInBytes() << ") "
		<< " getAlignment(" << layout->getAlignment() << ")"		
		<< " hasPadding(" << layout->hasPadding() << ")" << std::endl;
	for(int index=0; index < number_of_elements; ++index) {
		llvm::Type * et = structTy->getElementType(index);
		std::cout << "   element[" << index << "] offset in bytes = " << layout->getElementOffset(index) << 
				" type is "; 
				et->dump();
		std::cout << std::endl;
	}
		
}

void
LLVM_Util::validate_struct_data_layout(llvm::Type *Ty, const std::vector<unsigned int> & expected_offset_by_index)
{
	ASSERT(Ty);
	ASSERT(Ty->isStructTy());
			
	llvm::StructType *structTy = static_cast<llvm::StructType *>(Ty);
    const llvm::DataLayout & data_layout = m_llvm_exec->getDataLayout();
    
    int number_of_elements = structTy->getNumElements();


	const StructLayout * layout = data_layout.getStructLayout (structTy);
//	std::cout << "dump_struct_data_layout: getSizeInBytes(" << layout->getSizeInBytes() << ") "
//		<< " getAlignment(" << layout->getAlignment() << ")"		
//		<< " hasPadding(" << layout->hasPadding() << ")" << std::endl;
	
	for(int index=0; index < number_of_elements; ++index) {
		llvm::Type * et = structTy->getElementType(index);
		
		auto actual_offset = layout->getElementOffset(index);

		ASSERT(index < expected_offset_by_index.size());
		

		
//		std::cout << "   element[" << index << "] offset in bytes = " << actual_offset << " expect offset = " << expected_offset_by_index[index] << 
//				" type is "; 
//				et->dump();
				
				
		ASSERT(expected_offset_by_index[index] == actual_offset);
		std::cout << std::endl;
	}		
	if (expected_offset_by_index.size() != number_of_elements)
	{
		std::cout << "   expected " << expected_offset_by_index.size() << " members but actual member count is = " << number_of_elements << std::endl;
	}
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
    if (USE_MCJIT)
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
    //
    // LLVM keeps changing names and call sequence. This part is easier to
    // understand if we explicitly break it into individual LLVM versions.
#if OSL_LLVM_VERSION >= 37

    m_llvm_func_passes = new llvm::legacy::FunctionPassManager(module());
    llvm::legacy::FunctionPassManager &fpm = (*m_llvm_func_passes);

    m_llvm_module_passes = new llvm::legacy::PassManager;
    llvm::legacy::PassManager &mpm = (*m_llvm_module_passes);

#elif OSL_LLVM_VERSION >= 36

    m_llvm_func_passes = new llvm::legacy::FunctionPassManager(module());
    llvm::legacy::FunctionPassManager &fpm (*m_llvm_func_passes);
    fpm.add (new llvm::DataLayoutPass());

    m_llvm_module_passes = new llvm::legacy::PassManager;
    llvm::legacy::PassManager &mpm (*m_llvm_module_passes);
    mpm.add (new llvm::DataLayoutPass());

#elif OSL_LLVM_VERSION == 35

    m_llvm_func_passes = new llvm::legacy::FunctionPassManager(module());
    llvm::legacy::FunctionPassManager &fpm (*m_llvm_func_passes);
    fpm.add (new llvm::DataLayoutPass(module()));

    m_llvm_module_passes = new llvm::legacy::PassManager;
    llvm::legacy::PassManager &mpm (*m_llvm_module_passes);
    mpm.add (new llvm::DataLayoutPass(module()));

#elif OSL_LLVM_VERSION == 34

    m_llvm_func_passes = new llvm::legacy::FunctionPassManager(module());
    llvm::legacy::FunctionPassManager &fpm (*m_llvm_func_passes);
    fpm.add (new llvm::DataLayout(module()));

    m_llvm_module_passes = new llvm::legacy::PassManager;
    llvm::legacy::PassManager &mpm (*m_llvm_module_passes);
    mpm.add (new llvm::DataLayout(module()));

#endif

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
#if OSL_LLVM_VERSION <= 36
        // Is there a replacement for this in newer LLVM?
        mpm.add (llvm::createScalarReplAggregatesPass());
#endif
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

#if OSL_LLVM_VERSION > 35 && !defined(OSL_FORCE_BITCODE_PARSE)
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
#if OSL_LLVM_VERSION < 40
    for (llvm::Module::iterator iter = module()->begin(); iter != module()->end(); iter++) {
        llvm::Function *sym = static_cast<llvm::Function*>(iter);
#else
    for (llvm::Function& func : module()->getFunctionList()) {
        llvm::Function *sym = &func;
#endif
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
    if (fastcall) {
    	
    	std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>FAST_CALL MAKE FUNCTION=" << name << std::endl;
        func->setCallingConv(llvm::CallingConv::Fast);
    }
    return func;
}



llvm::Value *
LLVM_Util::current_function_arg (int a)
{
    llvm::Function::arg_iterator arg_it = current_function()->arg_begin();
    for (int i = 0;  i < a;  ++i)
        ++arg_it;
#if OSL_LLVM_VERSION <= 36
    return arg_it;
#else
    return &(*arg_it);
#endif
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
                        const std::string &name, bool is_packed)
{
	llvm::StructType * st = llvm::StructType::create(context(), types, name, is_packed);
	ASSERT(st->isStructTy());
	llvm::Type * t= st;
	ASSERT(t->isStructTy());
	return t;
    //return llvm::StructType::create(context(), types, name, is_packed);
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
	ASSERT(cv  != nullptr);
	return llvm::ConstantVector::getSplat(m_vector_width, cv); 
}


llvm::Value *
LLVM_Util::constant (float f)
{
    return llvm::ConstantFP::get (context(), llvm::APFloat(f));
}

llvm::Value *
LLVM_Util::wide_constant (float f)
{
	return llvm::ConstantVector::getSplat(m_vector_width, llvm::ConstantFP::get (context(), llvm::APFloat(f))); 
}

llvm::Value *
LLVM_Util::constant (int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(32,i));
}

llvm::Value *
LLVM_Util::constant64 (int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(64,i));
}

llvm::Value *
LLVM_Util::constant128 (int i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(128,i));
}

llvm::Value *
LLVM_Util::wide_constant (int i)
{
	return llvm::ConstantVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(32,i))); 
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
	return llvm::ConstantVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(bits,i))); 
}

llvm::Value *
LLVM_Util::constant_bool (bool i)
{
    return llvm::ConstantInt::get (context(), llvm::APInt(1,i));
}

llvm::Value *
LLVM_Util::wide_constant_bool (bool i)
{
	return llvm::ConstantVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(1,i))); 
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
    //return builder().CreateIntToPtr (str, type_string(), "ustring constant");
    
//    Value* emptyVec = UndefValue::get(type_wide_void_ptr());
    
    llvm::Value * constant_value = builder().CreateIntToPtr (str, type_string(), "ustring constant");
//    llvm::InsertElementInstr::Create(emptyVec, constant_value, llvm::ConstantInt::get (context(), llvm::APInt(32,i)));
    
    return builder().CreateVectorSplat(m_vector_width, constant_value);
//    
//    
//    return llvm::ConstantVector::getSplat(m_vector_width, llvm::ConstantInt::get (context(), llvm::APInt(32,i)));
}


llvm::Value *
LLVM_Util::mask_as_int(llvm::Value *mask)
{
    ASSERT(mask->getType() == type_wide_bool());

    llvm::Type * int_reinterpret_cast_vector_type = (llvm::Type *) llvm::Type::getInt16Ty (*m_llvm_context);

    llvm::Value* result = builder().CreateBitCast (mask, int_reinterpret_cast_vector_type);

    return builder().CreateZExt(result, (llvm::Type *) llvm::Type::getInt32Ty (*m_llvm_context));
}

llvm::Value *
LLVM_Util::int_as_mask(llvm::Value *value)
{
    ASSERT(value->getType() == type_int());

    llvm::Value* int16 = builder().CreateTrunc(value, (llvm::Type *) llvm::Type::getInt16Ty (*m_llvm_context));

    llvm::Value* result = builder().CreateBitCast (int16, type_wide_bool());

    ASSERT(result->getType() == type_wide_bool());

    return result;
}

llvm::Value *
LLVM_Util::test_if_mask_is_non_zero(llvm::Value *mask)
{
	ASSERT(mask->getType() == type_wide_bool());

	llvm::Type * extended_int_vector_type;
	llvm::Type * int_reinterpret_cast_vector_type;
	llvm::Value * zeroConstant;
	switch(m_vector_width) {
	case 4:
		extended_int_vector_type = (llvm::Type *) llvm::VectorType::get(llvm::Type::getInt32Ty (*m_llvm_context), m_vector_width);
		int_reinterpret_cast_vector_type = (llvm::Type *) llvm::Type::getInt128Ty (*m_llvm_context);
		zeroConstant = constant128(0);
		break;
	case 8:
		extended_int_vector_type = (llvm::Type *) llvm::VectorType::get(llvm::Type::getInt32Ty (*m_llvm_context), m_vector_width);
		int_reinterpret_cast_vector_type = (llvm::Type *) llvm::IntegerType::get(*m_llvm_context,256);
		zeroConstant = llvm::ConstantInt::get (context(), llvm::APInt(256,0));
		break;
	case 16:
		extended_int_vector_type = (llvm::Type *) llvm::VectorType::get(llvm::Type::getInt8Ty (*m_llvm_context), m_vector_width);
		int_reinterpret_cast_vector_type = (llvm::Type *) llvm::Type::getInt128Ty (*m_llvm_context);
		zeroConstant = constant128(0);
		break;
	default:
		ASSERT(0 && "Unhandled vector width");
		break;
	};		

	llvm::Value * wide_int_mask = builder().CreateSExt(mask, extended_int_vector_type);
	llvm::Value * mask_as_int =  builder().CreateBitCast (wide_int_mask, int_reinterpret_cast_vector_type);
    
    return op_ne (mask_as_int, zeroConstant);
}


llvm::Value *
LLVM_Util::widen_value (llvm::Value *val)
{
    return builder().CreateVectorSplat(m_vector_width, val);
}
 
llvm::Value * 
LLVM_Util::negate_mask(llvm::Value *mask)
{
	ASSERT(mask->getType() == type_wide_bool());
	return builder().CreateNot(mask);
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
        ASSERT (0 && "not handling this type yet");
    }
    if (typedesc.arraylen)
    {
    	
    	std::cout << "llvm_vector_type typedesc.arraylen = " << typedesc.arraylen << std::endl;
        lt = llvm::ArrayType::get (lt, typedesc.arraylen);
    }
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
LLVM_Util::wide_op_alloca (const TypeDesc &type, int n, const std::string &name)
{
    return op_alloca (llvm_vector_type(type.elementtype()), n*type.numelements(), name);
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



void
LLVM_Util::mark_structure_return_value(llvm::Value *funccall)
{
    llvm::CallInst* call = llvm::cast<llvm::CallInst>(funccall);
    
    auto attrs = llvm::AttributeSet::get(
    		call->getContext(),
        llvm::AttributeSet::FunctionIndex,
        llvm::Attribute::NoUnwind);

    //attrs = attrs.addAttribute(call->getContext(), 1, llvm::Attribute::NoAlias);

    attrs = attrs.addAttribute(call->getContext(), 1,
                               llvm::Attribute::StructRet);

    call->setAttributes(attrs);
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
#if OSL_LLVM_VERSION <= 36
    builder().CreateCall5 (func, ptr, fill_val, len, constant(align),
                           constant_bool(false));
#else
    llvm::Value *args[5] = {
        ptr, fill_val, len, constant(align), constant_bool(false)
    };
    builder().CreateCall (func, llvm::ArrayRef<llvm::Value*>(args, 5));

#endif
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
#if OSL_LLVM_VERSION <= 36
    builder().CreateCall5 (func, dst, src,
                           constant(len), constant(align), constant_bool(false));
#else
    llvm::Value *args[5] = {
        dst, src, constant(len), constant(align), constant_bool(false)
    };
    builder().CreateCall (func, llvm::ArrayRef<llvm::Value*>(args, 5));
#endif
}



llvm::Value *
LLVM_Util::op_load (llvm::Value *ptr)
{
    return builder().CreateLoad (ptr);
}


void
LLVM_Util::push_mask(llvm::Value *mask, bool negate, bool absolute)
{	
	ASSERT(mask->getType() == type_wide_bool());
	if(m_mask_stack.empty()) {
		m_mask_stack.push_back(MaskInfo{mask, negate});
	} else {
		
		MaskInfo & mi = m_mask_stack.back();
		llvm::Value *prev_mask = mi.mask;
		bool prev_negate = mi.negate;
	
		if (false == prev_negate) {
			if (false == negate)
			{
				llvm::Value *blended_mask;
				if (absolute) {
					blended_mask = mask;
				} else {
					blended_mask = builder().CreateSelect(prev_mask, mask, prev_mask);
				}
				m_mask_stack.push_back(MaskInfo{blended_mask, false});
			} else {				
				ASSERT(false == absolute);
				llvm::Value *blended_mask = builder().CreateSelect(mask, wide_constant_bool(false), prev_mask);
				m_mask_stack.push_back(MaskInfo{blended_mask, false});			
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
				m_mask_stack.push_back(MaskInfo{blended_mask, false});
			} else {
				ASSERT(false == absolute);
				llvm::Value *blended_mask = builder().CreateSelect(prev_mask, prev_mask, mask);
				m_mask_stack.push_back(MaskInfo{blended_mask, true});			
			}			
		}
	}
}

void
LLVM_Util::pop_if_mask()
{
	ASSERT(false == m_mask_stack.empty());
	
	if(m_mask_break_stack.empty()) {	
		m_mask_stack.pop_back();
	} else {
		m_mask_stack.pop_back();
		// Apply the break mask to the outter scope's mask (if one?)
		if (false == m_mask_stack.empty())
		{
			auto & mi = m_mask_stack.back();			
			llvm::Value * existing_mask = mi.mask;
			
			const auto & bsi = m_mask_break_stack.back();
			if (bsi.negate) {
				if(mi.negate) {
					mi.mask = builder().CreateSelect(bsi.mask, bsi.mask, existing_mask);
				} else {
					mi.mask = builder().CreateSelect(bsi.mask, wide_constant_bool(false), existing_mask);
				}
			} else {				
				if(mi.negate) {
					mi.mask = builder().CreateSelect(bsi.mask, existing_mask, wide_constant_bool(true));
				} else {
					mi.mask = builder().CreateSelect(bsi.mask, existing_mask, bsi.mask);
				}
			}			
		}		
	}
}

void
LLVM_Util::pop_loop_mask()
{
	ASSERT(false == m_mask_stack.empty());
	m_mask_stack.pop_back();
}

llvm::Value *
LLVM_Util::current_mask()
{
	if(m_mask_stack.empty()) {
		return wide_constant_bool(true);
	} else {
		auto & mi = m_mask_stack.back();
		if (mi.negate) {
			llvm::Value *negated_mask = builder().CreateSelect(mi.mask, wide_constant_bool(false), wide_constant_bool(true));
			return negated_mask;
		} else {
			return mi.mask;
		}
	}
}

llvm::Value *
LLVM_Util::apply_break_mask_to(llvm::Value *existing_mask)
{
	if(m_mask_break_stack.empty()) {
		return existing_mask;
	} else {
		auto & bsi = m_mask_break_stack.back();
		if (bsi.negate) {
			llvm::Value *result = builder().CreateSelect(bsi.mask, wide_constant_bool(false), existing_mask);
			return result;
		} else {
			llvm::Value *result = builder().CreateSelect(bsi.mask, existing_mask, bsi.mask);
			return result;
		}
	}
}

void
LLVM_Util::push_mask_break()
{
	ASSERT(false == m_mask_stack.empty());

	// TODO: determine if we need a stack or just the latest break
	{
		MaskInfo copy_of_mi = m_mask_stack.back();
		copy_of_mi.negate = !copy_of_mi.negate;
		m_mask_break_stack.push_back(copy_of_mi);
	}
		
	// Now modify the current mask to turn off all lanes
	// because the only active lanes just hit a break statement
	// so all future instructions should execute against an empty mask
	// NOTE: this is technically unreachable code, ideally front end
	// optimizations would get rid of it before hand
	// at this point don't want to introduce complexity of trying to
	// skip instructions
	auto & mi = m_mask_stack.back();
	mi.mask = wide_constant_bool(false);
	// NOTE: if there are no other instructions, then this mask will just not
	// get used/generated (we think)
}

void
LLVM_Util::clear_mask_break()
{
	m_mask_break_stack.clear();
}


void
LLVM_Util::push_masking_enabled(bool enabled)
{
	m_enable_masking_stack.push_back(enabled);	
}

void
LLVM_Util::pop_masking_enabled()
{
	ASSERT(false == m_enable_masking_stack.empty());
	m_enable_masking_stack.pop_back();	
}



void
LLVM_Util::op_store (llvm::Value *val, llvm::Value *ptr)
{	
	if(m_mask_stack.empty() || val->getType()->isVectorTy() == false || m_enable_masking_stack.empty() || m_enable_masking_stack.back() == false) {		
		// We may not be in a non-uniform code block
		// or the value being stored may be uniform, which case it shouldn't
		// be a vector type
	    builder().CreateStore (val, ptr);		
	} else {				
		// TODO: could probably make these DASSERT as  the conditional above "should" be checking all of this
		ASSERT(m_enable_masking_stack.back());
		ASSERT(val->getType()->isVectorTy());
		ASSERT(false == m_mask_stack.empty());
		
		MaskInfo & mi = m_mask_stack.back();
		// TODO: add assert for ptr alignment in debug builds	
#if 0
		if (m_supports_masked_stores) {
			builder().CreateMaskedStore(val, ptr, 64, mi.mask);
		} else 
#endif
		{
			// Transform the masted store to a load+blend+store
			// Technically, the behavior is different than a masked store
			// as different thread could technically have modified the masked off
			// data lane values inbetween the read+store
			// As this language sits below the threading level that could
			// never happen and a read+store
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
#if OSL_LLVM_VERSION <= 36
    return builder().CreateConstGEP2_32 (ptr, elem1, elem2);
#else
    return builder().CreateConstGEP2_32 (nullptr, ptr, elem1, elem2);
#endif
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
        
    ASSERT (0 && "Op has bad value type combination");
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
        
    ASSERT (0 && "Op has bad value type combination");
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
    
    ASSERT (0 && "Op has bad value type combination");
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
           
    ASSERT (0 && "Op has bad value type combination");
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
        
    ASSERT (0 && "Op has bad value type combination");
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

    ASSERT (0 && "Op has bad value type combination");
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
    ASSERT (0 && "Op has bad value type combination");
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
    
    ASSERT (0 && "Op has bad value type combination");
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
    ASSERT (0 && "Op has bad value type combination");
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
    ASSERT (0 && "Op has bad value type combination");
}


llvm::Value *
LLVM_Util::op_int_to_bool(llvm::Value* a)
{
    if (a->getType() == type_int()) 
    	return builder().CreateTrunc (a, type_bool());
    if (a->getType() == type_wide_int()) 
    	return builder().CreateTrunc (a, type_wide_bool());
    if ((a->getType() == type_bool()) || (a->getType() == type_wide_bool()))
        return a;
    ASSERT (0 && "Op has bad value type combination");
	return NULL;
}


llvm::Value *
LLVM_Util::op_and (llvm::Value *a, llvm::Value *b)
{
	// TODO: unlclear why inconsistent and not checking for operand types 
	// with final ASSERT for "bad value type combination"
    return builder().CreateAnd (a, b);
}


llvm::Value *
LLVM_Util::op_or (llvm::Value *a, llvm::Value *b)
{
	// TODO: unlclear why inconsistent and not checking for operand types 
	// with final ASSERT for "bad value type combination"
    return builder().CreateOr (a, b);
}


llvm::Value *
LLVM_Util::op_xor (llvm::Value *a, llvm::Value *b)
{
	// TODO: unlclear why inconsistent and not checking for operand types 
	// with final ASSERT for "bad value type combination"
    return builder().CreateXor (a, b);
}


llvm::Value *
LLVM_Util::op_shl (llvm::Value *a, llvm::Value *b)
{
	// TODO: unlclear why inconsistent and not checking for operand types 
	// with final ASSERT for "bad value type combination"
    return builder().CreateShl (a, b);
}


llvm::Value *
LLVM_Util::op_shr (llvm::Value *a, llvm::Value *b)
{
    if ((a->getType() == type_int() && b->getType() == type_int()) ||
		(a->getType() == type_wide_int() && b->getType() == type_wide_int()))
        return builder().CreateAShr (a, b);  // signed int -> arithmetic shift
    
    ASSERT (0 && "Op has bad value type combination");
}


llvm::Value *
LLVM_Util::op_not (llvm::Value *a)
{
	// TODO: unlclear why inconsistent and not checking for operand types 
	// with final ASSERT for "bad value type combination"
    return builder().CreateNot (a);
}



llvm::Value *
LLVM_Util::op_select (llvm::Value *cond, llvm::Value *a, llvm::Value *b)
{
	// TODO: unlclear why inconsistent and not checking for operand types 
	// with final ASSERT for "bad value type combination"
    return builder().CreateSelect (cond, a, b);
}

llvm::Value *
LLVM_Util::op_extract (llvm::Value *a, int index)
{
    return builder().CreateExtractElement (a, index);
}

llvm::Value *
LLVM_Util::op_eq (llvm::Value *a, llvm::Value *b, bool ordered)
{
    if (a->getType() != b->getType()) {
    	std::cout << "a type=" << llvm_typenameof(a) << " b type=" << llvm_typenameof(b) << std::endl;
    }
    ASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOEQ (a, b) : builder().CreateFCmpUEQ (a, b);
    else
        return builder().CreateICmpEQ (a, b);
}



llvm::Value *
LLVM_Util::op_ne (llvm::Value *a, llvm::Value *b, bool ordered)
{
    if (a->getType() != b->getType()) {
    	std::cout << "a type=" << llvm_typenameof(a) << " b type=" << llvm_typenameof(b) << std::endl;
    }
    ASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpONE (a, b) : builder().CreateFCmpUNE (a, b);
    else
        return builder().CreateICmpNE (a, b);
}



llvm::Value *
LLVM_Util::op_gt (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOGT (a, b) : builder().CreateFCmpUGT (a, b);
    else
        return builder().CreateICmpSGT (a, b);
}



llvm::Value *
LLVM_Util::op_lt (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOLT (a, b) : builder().CreateFCmpULT (a, b);
    else
        return builder().CreateICmpSLT (a, b);
}



llvm::Value *
LLVM_Util::op_ge (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOGE (a, b) : builder().CreateFCmpUGE (a, b);
    else
        return builder().CreateICmpSGE (a, b);
}



llvm::Value *
LLVM_Util::op_le (llvm::Value *a, llvm::Value *b, bool ordered)
{
    ASSERT (a->getType() == b->getType());
    if ((a->getType() == type_float()) || (a->getType() == type_wide_float()))
        return ordered ? builder().CreateFCmpOLE (a, b) : builder().CreateFCmpULE (a, b);
    else
        return builder().CreateICmpSLE (a, b);
}



void
LLVM_Util::write_bitcode_file (const char *filename, std::string *err)
{
#if OSL_LLVM_VERSION >= 36
    std::error_code local_error;
    llvm::raw_fd_ostream out (filename, local_error, llvm::sys::fs::F_None);
#elif OSL_LLVM_VERSION >= 35
    std::string local_error;
    llvm::raw_fd_ostream out (filename, err ? *err : local_error, llvm::sys::fs::F_None);
#else
    std::string local_error;
    llvm::raw_fd_ostream out (filename, err ? *err : local_error);
#endif
    llvm::WriteBitcodeToFile (module(), out);

#if OSL_LLVM_VERSION >= 36
    if (err && local_error)
        *err = local_error.message ();
#endif
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
