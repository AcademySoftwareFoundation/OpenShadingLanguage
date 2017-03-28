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
#include <stack>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>

#include "oslexec_pvt.h"
#include "backendllvm_wide.h"

#include <llvm/IR/Type.h>

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



BackendLLVMWide::BackendLLVMWide (ShadingSystemImpl &shadingsys,
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
}



BackendLLVMWide::~BackendLLVMWide ()
{
}



int
BackendLLVMWide::llvm_debug() const
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
BackendLLVMWide::set_inst (int layer)
{
    OSOProcessorBase::set_inst (layer);  // parent does the heavy lifting
    ll.debug (llvm_debug());
}



llvm::Type *
BackendLLVMWide::llvm_pass_type (const TypeSpec &typespec)
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

llvm::Type *
BackendLLVMWide::llvm_pass_wide_type (const TypeSpec &typespec)
{
    if (typespec.is_closure_based())
        return (llvm::Type *) ll.type_void_ptr();
    TypeDesc t = typespec.simpletype().elementtype();
    llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = (llvm::Type *)ll.type_void_ptr(); // ll.type_wide_float();
    else if (t == TypeDesc::INT)
        lt = (llvm::Type *)ll.type_void_ptr(); // ll.type_wide_int();
    else if (t == TypeDesc::STRING)
        lt = (llvm::Type *)ll.type_void_ptr(); // (llvm::Type *) ll.type_wide_ string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = (llvm::Type *) ll.type_void_ptr(); //llvm_type_wide_triple_ptr();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = (llvm::Type *) ll.type_void_ptr(); //llvm_type_wide_matrix_ptr();
    else if (t == TypeDesc::NONE)
        lt = ll.type_void();
    else if (t == TypeDesc::PTR)
        lt = (llvm::Type *) ll.type_void_ptr();
    else if (t == TypeDesc::LONGLONG)
        lt = (llvm::Type *)ll.type_void_ptr(); // ll.type_wide_longlong();
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
BackendLLVMWide::llvm_assign_zero (const Symbol &sym)
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
BackendLLVMWide::llvm_zero_derivs (const Symbol &sym)
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
BackendLLVMWide::llvm_zero_derivs (const Symbol &sym, llvm::Value *count)
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
    // ShaderGlobalsBatch struct in ShaderGlobals.h, as well as the llvm 'sg' type
    // defined in llvm_type_sg().
    static ustring fields[] = {
		// Uniform
		ustring("renderstate"),
		ustring("tracedata"), 
		ustring("objdata"),
		ustring("shadingcontext"), 
        ustring("renderer"),
        ustring("Ci"),
		ustring("raytype"),
		ustring("pad0"),
		ustring("pad1"),
		ustring("pad2"),
		// Varying
        ustring("P"), 
		ustring("dPdz"), 
		ustring("I"),
        ustring("N"), 
		ustring("Ng"),
        ustring("u"), 
		ustring("v"), 
		ustring("dPdu"), 
		ustring("dPdv"),
        ustring("time"), 
		ustring("dtime"), 
		ustring("dPdtime"), 
		ustring("Ps"),        
        ustring("object2common"), 
		ustring("shader2common"),
        ustring("surfacearea"), 
        ustring("flipHandedness"), 
		ustring("backfacing")
    };

    static bool field_is_uniform[] = {
		// Uniform
		true, //ustring("renderstate"),
		true, //ustring("tracedata"), 
		true, //ustring("objdata"),
		true, //ustring("shadingcontext"), 
		true, //ustring("renderer"),
		true, //ustring("Ci"),
		true, //ustring("raytype"),
		true, //ustring("pad0"),
		true, //ustring("pad1"),
		true, //ustring("pad2"),
		// Varying
        false, //ustring("P"), 
		false, //ustring("dPdz"), 
		false, //ustring("I"),
		false, //ustring("N"), 
		false, //ustring("Ng"),
		false, //ustring("u"), 
		false, //ustring("v"), 
		false, //ustring("dPdu"), 
		false, //ustring("dPdv"),
		false, //ustring("time"), 
		false, //ustring("dtime"), 
		false, //ustring("dPdtime"), 
		false, //ustring("Ps"),        
		false, //ustring("object2common"), 
		false, //ustring("shader2common"),
		false, //ustring("surfacearea"), 
		false, //ustring("flipHandedness"), 
		false, //ustring("backfacing")
    };
    
    
    bool
    IsShaderGlobalUniformByName (ustring name)
    {
        for (int i = 0;  i < int(sizeof(fields)/sizeof(fields[0]));  ++i) {
            if (name == fields[i]) {
            	return field_is_uniform[i];
            }
        }
        return false;
    }
    
}

int
BackendLLVMWide::ShaderGlobalNameToIndex (ustring name, bool &is_uniform)
{
    for (int i = 0;  i < int(sizeof(fields)/sizeof(fields[0]));  ++i)
        if (name == fields[i]) {
        	is_uniform = field_is_uniform[i];
            return i;
        }
    std::cout << "ShaderGlobalNameToIndex failed with " << name << std::endl;
    return -1;
}



llvm::Value *
BackendLLVMWide::llvm_global_symbol_ptr (ustring name, bool &is_uniform)
{
    // Special case for globals -- they live in the ShaderGlobals struct,
    // we use the name of the global to find the index of the field within
    // the ShaderGlobals struct.
    int sg_index = ShaderGlobalNameToIndex (name, is_uniform);
    ASSERT (sg_index >= 0);
    return ll.void_ptr (ll.GEP (sg_ptr(), 0, sg_index));
}

llvm::Value *
BackendLLVMWide::getLLVMSymbolBase (const Symbol &sym)
{
    Symbol* dealiased = sym.dealias();

	bool is_uniform = isSymbolUniform(sym);
	
    if (sym.symtype() == SymTypeGlobal) {
        llvm::Value *result = llvm_global_symbol_ptr (sym.name(), is_uniform);
        ASSERT (result);
        if (is_uniform) {
        	result = ll.ptr_to_cast (result, llvm_type(sym.typespec().elementtype()));
        } else {
        	result = ll.ptr_to_cast (result, llvm_wide_type(sym.typespec().elementtype()));
        }
        return result;
    }

    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        // Special case for params -- they live in the group data
        int fieldnum = m_param_order_map[&sym];
        return groupdata_field_ptr (fieldnum, sym.typespec().elementtype().simpletype(), is_uniform);
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


bool 
BackendLLVMWide::isSymbolUniform(const Symbol& sym)
{
	if (m_is_uniform_by_symbol.size() == 0) {
		
		const OpcodeVec & opcodes = inst()->ops();
		
		ASSERT(m_requires_masking_by_op_index.empty());
		m_requires_masking_by_op_index.resize(opcodes.size(), false);
				
		// TODO:  Optimize: could probably use symbol index vs. a pointer 
		// allowing a lookup table vs. hash_map
		 
		
		std::unordered_multimap<const Symbol * /* parent */ , const Symbol * /* dependent */> symbolFeedForwardMap;

		struct UsageInfo
		{
			int last_depth;
			int last_maskId;
			std::vector <std::pair<int /*blockDepth*/, int /*op_num*/>> potentially_unmasked_ops;
		};
		std::unordered_map<const Symbol *, UsageInfo > usageInfoBySymbol;
		
		std::vector<const Symbol *> symbolsCurrentBlockDependsOn;
		
		int nextMaskId = 0;
		int blockId = 0;
    	std::function<void(int, int, int, int)> discoverSymbolsBetween;
    	discoverSymbolsBetween = [&](int beginop, int endop, int blockDepth, int maskId)->void
		{		
			for(int opIndex = beginop; opIndex < endop; ++opIndex)
			{
				Opcode & opcode = op(opIndex);
				std::cout << "op=" << opcode.opname();
				int argCount = opcode.nargs();
				
				const Symbol * symbolsReadByOp[argCount];
				int symbolsRead = 0;
				const Symbol * symbolsWrittenByOp[argCount];
				int symbolsWritten = 0;
				for(int argIndex = 0; argIndex < argCount; ++argIndex) {
					const Symbol * aSymbol = opargsym (opcode, argIndex);
					if (opcode.argwrite(argIndex)) {
						std::cout << " write to ";
						symbolsWrittenByOp[symbolsWritten++] = aSymbol;
					}
					if (opcode.argread(argIndex)) {
						symbolsReadByOp[symbolsRead++] = aSymbol;
						std::cout << " read from ";					
					}
					std::cout << " " << aSymbol->name();
					
					bool isUniform = true;
					if (aSymbol->symtype() == SymTypeOutputParam) {
						
							//&& ! aSymbol->lockgeom() && ! aSymbol->typespec().is_closure()
							//&& ! aSymbol->connected() && ! aSymbol->connected_down())
						isUniform = false;
					}
					std::cout << " discovery " << aSymbol->name() << " initial isUniform=" << isUniform << std::endl;
					m_is_uniform_by_symbol[aSymbol] = isUniform;
				}
				std::cout << std::endl;
				
				for(int readIndex=0; readIndex < symbolsRead; ++readIndex) {
					const Symbol * symbolReadFrom = symbolsReadByOp[readIndex];
					for(int writeIndex=0; writeIndex < symbolsWritten; ++writeIndex) {
						const Symbol * symbolWrittenTo = symbolsWrittenByOp[writeIndex];
						// Skip self dependencies
						if (symbolWrittenTo != symbolReadFrom) {
							symbolFeedForwardMap.insert(std::make_pair(symbolReadFrom, symbolWrittenTo));
						}
					}		
					
					// Check if reading a Symbol that was written to from a different 
					// maskId than we are reading, if so we need to mark it as requiring masking
					auto lookup = usageInfoBySymbol.find(symbolReadFrom);
					if(lookup != usageInfoBySymbol.end()) {
						UsageInfo & info = lookup->second;
						if ((info.last_depth > blockDepth) && (info.last_maskId != maskId))
						{
							std::cout << symbolReadFrom->name() << " will need to have last write be masked" << std::endl;
							ASSERT(info.potentially_unmasked_ops.empty() == false);
							decltype(info.potentially_unmasked_ops) remaining_ops;
							for(auto usage: info.potentially_unmasked_ops) {
								// Only mark deeper usages as requiring masking
								if(usage.first > blockDepth)
								{
									std::cout << " marking op " << usage.second << " as masked" << std::endl;
									m_requires_masking_by_op_index[usage.second] = true;									
								} else {
									remaining_ops.push_back(usage);
								}
							}
							
							info.potentially_unmasked_ops.swap(remaining_ops);		
							// Now that all ops writing to the symbol at higher depths have been marked to be masked
							// we can now consider the matter handled at this point at reset the
							// last_depth written at to the current depth to avoid needlessly repeating the work.
							info.last_depth = blockDepth;
						}
					}
				}
				
				for(int writeIndex=0; writeIndex < symbolsWritten; ++writeIndex) {
					const Symbol * symbolWrittenTo = symbolsWrittenByOp[writeIndex];
					UsageInfo & info = usageInfoBySymbol[symbolWrittenTo];
					info.last_depth = blockDepth;
					info.last_maskId = maskId;
					info.potentially_unmasked_ops.push_back(std::make_pair(blockDepth, opIndex));
				}
				
				// Add dependencies between symbols written to in this basic block
				// to the set of symbols the code blocks where dependent upon to be executed
				for(const Symbol *symbolCurrentBlockDependsOn : symbolsCurrentBlockDependsOn)
				{
					for(int writeIndex=0; writeIndex < symbolsWritten; ++writeIndex) {
						const Symbol * symbolWrittenTo = symbolsWrittenByOp[writeIndex];
						// Skip self dependencies
						if (symbolWrittenTo != symbolCurrentBlockDependsOn) {
							symbolFeedForwardMap.insert(std::make_pair(symbolCurrentBlockDependsOn, symbolWrittenTo));
						}
					}									
				}
				
				if (opcode.jump(0) >= 0)
				{
					// The operation with a jump depends on reading the follow symbols
					// track them for the following basic blocks as the writes
					// within those basic blocks will depend on the uniformity of 
					// the values read by this operation
			    	std::function<void()> pushSymbolsCurentBlockDependsOn;
			    	pushSymbolsCurentBlockDependsOn = [&]()->void {			    			
						for(int readIndex=0; readIndex < symbolsRead; ++readIndex) {
							const Symbol * symbolReadFrom = symbolsReadByOp[readIndex];
							symbolsCurrentBlockDependsOn.push_back(symbolReadFrom);
						}
			    	};
									
					// op must have jumps, therefore have nested code we need to process
					// We need to process these in the same order as the code generator
					// so our "block depth" lines up for symbol lookups
					if (opcode.opname() == ustring("if"))
					{
						pushSymbolsCurentBlockDependsOn();
						// Then block
						discoverSymbolsBetween(opIndex+1, opcode.jump(0), blockDepth+1, nextMaskId++);
						// else block
						discoverSymbolsBetween(opcode.jump(0), opcode.jump(1), blockDepth+1, nextMaskId++);
					} else if (opcode.opname() == ustring("for"))
					{
						// Init block
						// NOTE: init block doesn't depend on the for loops conditions and should be exempt
						discoverSymbolsBetween(opIndex+1, opcode.jump(0), blockDepth, maskId);						
						// Condition block
						// NOTE: the first execution of the condition doesn't depend on the for loops conditions and should be exempt
						// TODO: unclear about subsequent executions, they might need to be masked... Hmmm
						discoverSymbolsBetween(opcode.jump(0), opcode.jump(1), blockDepth, maskId);
						
						pushSymbolsCurentBlockDependsOn();
						
						int maskIdForBodyAndStep = nextMaskId++;
						// Body block
						discoverSymbolsBetween(opcode.jump(1), opcode.jump(2), blockDepth+1, maskIdForBodyAndStep);
						// Step block
						discoverSymbolsBetween(opcode.jump(2), opcode.jump(3), blockDepth+1, maskIdForBodyAndStep);
					} else {

						ASSERT(0 && "Unhandled OSL instruction which contains jumps, note this uniform detection code needs to walk the code blocks identical to build_llvm_code");
					}

					// Now that we have processed the dependent basic blocks
					// we continue processing instructions and those will no
					// longer be dependent on this operations read symbols
					for(int readIndex=symbolsRead-1; readIndex >= 0; --readIndex) {
						// TODO: change to DASSERT later once we are confident
						ASSERT(symbolsCurrentBlockDependsOn.back() == symbolsReadByOp[readIndex]);
						symbolsCurrentBlockDependsOn.pop_back();
					}					
				}
				
		        // If the op we coded jumps around, skip past its recursive block
		        // executions.
		        int next = opcode.farthest_jump ();
		        if (next >= 0)
		        	opIndex = next-1;				
			}
		};
    	
    	discoverSymbolsBetween(inst()->maincodebegin(), inst()->maincodeend(), 0, nextMaskId++);
    	
		std::cout << "About to build m_is_uniform_by_symbol" << std::endl;			
		
    	std::function<void(const Symbol *)> recursivelyMarkNonUniform;
    	recursivelyMarkNonUniform = [&](const Symbol* nonUniformSymbol)->void
		{
        	bool previously_was_uniform = m_is_uniform_by_symbol[nonUniformSymbol];
        	m_is_uniform_by_symbol[nonUniformSymbol] = false;
        	if (previously_was_uniform) {
        		auto range = symbolFeedForwardMap.equal_range(nonUniformSymbol);
        		auto iter = range.first;
        		for(;iter != range.second; ++iter) {
        			const Symbol * symbolWrittenTo = iter->second;
        			recursivelyMarkNonUniform(symbolWrittenTo);
        		};
        	}
		};
		
		auto endOfFeeds = symbolFeedForwardMap.end();	
		for(auto feedIter = symbolFeedForwardMap.begin();feedIter != endOfFeeds; )
		{
			const Symbol * symbolReadFrom = feedIter->first;
			//std::cout << " " << symbolReadFrom->name() << " feeds into " << symbolWrittenTo->name() << std::endl;
			
			bool is_uniform = true;			
			auto symType = symbolReadFrom->symtype();
			if (symType == SymTypeGlobal) {
				is_uniform = IsShaderGlobalUniformByName(symbolReadFrom->name());
			} else if (symType == SymTypeParam) {
					// TODO: perhaps the connected params do not necessarily
					// need to be varying 
					is_uniform = false;
			}
			if (is_uniform == false) {
				// So we have a symbol that is not uniform, so it will be a wide type
				// Thus anyone who depends on it will need to be wide as well.
				recursivelyMarkNonUniform(symbolReadFrom);
			}
			
			// The multimap may have multiple entries with the same key
			// And we only need to iterate over unique keys,
			// so skip any consecutive duplicate keys
			do {
				++feedIter;
			} while (feedIter != endOfFeeds && symbolReadFrom == feedIter->first);
		}

		std::cout << "Emit m_is_uniform_by_symbol" << std::endl;			
		
		for(auto rIter = m_is_uniform_by_symbol.begin(); rIter != m_is_uniform_by_symbol.end(); ++rIter) {
			const Symbol * rSym = rIter->first;
			bool is_uniform = rIter->second;
			std::cout << "--->" << rSym << " " << rSym->name() << " is " << (is_uniform ? "UNIFORM" : "VARYING") << std::endl;			
		}
		std::cout << std::flush;		
	}
	
	
	auto iter = m_is_uniform_by_symbol.find(&sym);
	if (iter == m_is_uniform_by_symbol.end()) 
	{	// TODO:  Any symbols not involved in oprations would be uniform
		// unless they are an output, but I think not just an output of an invidual
		// shader, but the output of the entire network
		std::cout << " undiscovered " << sym.name() << " initial isUniform=";
		if (sym.symtype() == SymTypeOutputParam) {
                //&& ! sym.lockgeom() && ! sym.typespec().is_closure()
                //&& ! sym.connected() && ! sym.connected_down())
			std::cout << false << std::endl;
			return false;
		}
		std::cout << true << std::endl;		
		return true;
	}
	
	bool is_uniform = iter->second;
	return is_uniform;
}

bool 
BackendLLVMWide::requiresMasking(int opIndex)
{
	ASSERT(m_requires_masking_by_op_index.empty() == false);
	ASSERT(m_requires_masking_by_op_index.size() > opIndex);
	return m_requires_masking_by_op_index[opIndex];
}


llvm::Value *
BackendLLVMWide::llvm_alloca (const TypeSpec &type, bool derivs, bool is_uniform,
                          const std::string &name)
{
	std::cout << "llvm_alloca " << name ;
    TypeDesc t = llvm_typedesc (type);
    int n = derivs ? 3 : 1;
    std::cout << "n=" << n << " t.size()=" << t.size();
    m_llvm_local_mem += t.size() * n;
    if (is_uniform)
    {
    	std::cout << " as UNIFORM " << std::endl ;
        return ll.op_alloca (t, n, name);
    } else {
    	std::cout << " as VARYING " << std::endl ;
    	return ll.wide_op_alloca (t, n, name);
    }
}



llvm::Value *
BackendLLVMWide::getOrAllocateLLVMSymbol (const Symbol& sym)
{
    DASSERT ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp ||
              sym.symtype() == SymTypeConst)
             && "getOrAllocateLLVMSymbol should only be for local, tmp, const");
    Symbol* dealiased = sym.dealias();
    std::string mangled_name = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);

    if (map_iter == named_values().end()) {
    	bool is_uniform = isSymbolUniform(sym);
    	
        llvm::Value* a = llvm_alloca (sym.typespec(), sym.has_derivs(), is_uniform, mangled_name);
        named_values()[mangled_name] = a;
        return a;
    }
    return map_iter->second;
}



llvm::Value *
BackendLLVMWide::llvm_get_pointer (const Symbol& sym, int deriv,
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
        						// Constants by definition should always be UNIFORM
        						ll.type_ptr (llvm_type(sym.typespec().elementtype())));

    } else {
        // Start with the initial pointer to the variable's memory location
        result = getLLVMSymbolBase (sym);
    	std::cerr << " llvm_get_pointer(" << sym.name() << ") result=";
    	ll.llvm_typeof(result)->dump();
    	std::cerr << std::endl;
        
    }
    if (!result)
        return NULL;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t = sym.typespec().simpletype();
    if (t.arraylen || has_derivs) {
    	std::cout << "llvm_get_pointer we're dealing with derivatives<-------" << std::endl;
    	std::cout << "arrayindex" << arrayindex << "deriv=" << deriv << " t.arraylen="  << t.arraylen << std::endl;
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
BackendLLVMWide::llvm_load_value (const Symbol& sym, int deriv,
                                   llvm::Value *arrayindex, int component,
                                   TypeDesc cast, bool op_is_uniform)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
    	if (op_is_uniform) {
    		return ll.constant (0.0f);
    	} else {
    		return ll.wide_constant (0.0f);
    	}
    	
    }

    // arrayindex should be non-NULL if and only if sym is an array
    ASSERT (sym.typespec().is_array() == (arrayindex != NULL));

    if (sym.is_constant() && !sym.typespec().is_array() && !arrayindex) {
        // Shortcut for simple constants
        if (sym.typespec().is_float()) {
            if (cast == TypeDesc::TypeInt)
            	if (op_is_uniform) {
            		return ll.constant ((int)*(float *)sym.data());
            	} else {
                    return ll.wide_constant ((int)*(float *)sym.data());            		
            	}
            else
            	if (op_is_uniform) {
            		return ll.constant (*(float *)sym.data());
            	} else {
                    return ll.wide_constant (*(float *)sym.data());            		
            	}
        }
        if (sym.typespec().is_int()) {
            if (cast == TypeDesc::TypeFloat)
            	if (op_is_uniform) {
            		return ll.constant ((float)*(int *)sym.data());
            	} else {
                    return ll.wide_constant ((float)*(int *)sym.data());            		
            	}
            else
            {
            	if (op_is_uniform) {
            		return ll.constant (*(int *)sym.data());
            	} else {
                    return ll.wide_constant (*(int *)sym.data());            		
            	}
            }
        }
        if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
        	if (op_is_uniform) {
        		return ll.constant (((float *)sym.data())[component]);
        	} else {
                return ll.wide_constant (((float *)sym.data())[component]);        		
        	}
        }
        if (sym.typespec().is_string()) {
			// TODO:  NOT SURE WHAT TO DO WITH VARYING STRING
            //return ll.wide_constant (*(ustring *)sym.data());
        	ASSERT(op_is_uniform);
            return ll.constant (*(ustring *)sym.data());
        }
        ASSERT (0 && "unhandled constant type");
    }

    std::cout << "  llvm_load_value " << sym.typespec().string() << " cast " << cast << std::endl;
    return llvm_load_value (llvm_get_pointer (sym), sym.typespec(),
                            deriv, arrayindex, component, cast, op_is_uniform);
}



llvm::Value *
BackendLLVMWide::llvm_load_value (llvm::Value *ptr, const TypeSpec &type,
                                   int deriv, llvm::Value *arrayindex,
                                   int component, TypeDesc cast, bool op_is_uniform)
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
    {
    	std::cout << "step to the right field" << std::endl;
        ptr = ll.GEP (ptr, 0, component);
    }

    // Now grab the value
    llvm::Value *result = ll.op_load (ptr);

    if (type.is_closure_based())
        return result;

    // Handle int<->float type casting
    if (op_is_uniform) {
		if (type.is_floatbased() && cast == TypeDesc::TypeInt)
			result = ll.op_float_to_int (result);
		else if (type.is_int() && cast == TypeDesc::TypeFloat)
			result = ll.op_int_to_float (result);
    } else {
    	// TODO:  remove this assert once we have confirmed correct handling off all the
    	// different data types.  Using assert as a checklist to verify what we have 
    	// handled so far during development
    	ASSERT(cast == TypeDesc::UNKNOWN || cast == TypeDesc::TypeColor || cast == TypeDesc::TypeVector || cast == TypeDesc::TypePoint || cast == TypeDesc::TypeFloat || cast == TypeDesc::TypeInt);
    	
		if (type.is_floatbased() && cast == TypeDesc::TypeInt)
			result = ll.wide_op_float_to_int (result);
		else if (type.is_int() && cast == TypeDesc::TypeFloat)
			result = ll.wide_op_int_to_float (result);
    	
    	if (ll.llvm_typeof(result) ==  ll.type_float()) {
            result = ll.widen_value(result);    		    		
    	} else if (ll.llvm_typeof(result) ==  ll.type_triple()) {
            result = ll.widen_value(result);    		    		
    	} else if (ll.llvm_typeof(result) ==  ll.type_int()) {
            result = ll.widen_value(result);    		    		
    	} else {
        	ASSERT((ll.llvm_typeof(result) ==  ll.type_wide_float()) ||
        		   (ll.llvm_typeof(result) ==  ll.type_wide_int()) ||
        		   (ll.llvm_typeof(result) ==  ll.type_wide_triple()));
    	}
    }

    return result;
}



llvm::Value *
BackendLLVMWide::llvm_load_constant_value (const Symbol& sym, 
                                       int arrayindex, int component,
                                       TypeDesc cast,
									   bool op_is_uniform)
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
        	if (op_is_uniform) {
        		return ll.constant ((int)val[arrayindex]);
        	} else {
        		return ll.wide_constant ((int)val[arrayindex]);        		
        	}
        else
        	if (op_is_uniform) {
        		return ll.constant (val[arrayindex]);
        	} else
        	{
                return ll.wide_constant (val[arrayindex]);        		
        	}
    }
    if (sym.typespec().is_int()) {
        const int *val = (const int *)sym.data();
        if (cast == TypeDesc::TypeFloat)
        	if (op_is_uniform) {
        		return ll.constant ((float)val[arrayindex]);
        	} else {
                return ll.wide_constant ((float)val[arrayindex]);        		
        	}
        else
        	if (op_is_uniform) {
        		return ll.constant (val[arrayindex]);
        	} else {
                return ll.wide_constant (val[arrayindex]);        		
        	}
    }
    if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
        const float *val = (const float *)sym.data();
        int ncomps = (int) sym.typespec().aggregate();
    	if (op_is_uniform) {
    		return ll.constant (val[ncomps*arrayindex + component]);
    	} else {
            return ll.wide_constant (val[ncomps*arrayindex + component]);
    	}
    }
    if (sym.typespec().is_string()) {
        const ustring *val = (const ustring *)sym.data();
    	if (op_is_uniform) {
    		return ll.constant (val[arrayindex]);
    	} else {
            return ll.wide_constant (val[arrayindex]);    		
    	}
    }

    ASSERT (0 && "unhandled constant type");
    return NULL;
}



llvm::Value *
BackendLLVMWide::llvm_load_component_value (const Symbol& sym, int deriv,
                                             llvm::Value *component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        ASSERT (sym.typespec().is_floatbased() && 
                "can't ask for derivs of an int");
		// TODO:  switching back to non-wide to figure out uniform vs. varying data
        //return ll.wide_constant (0.0f);
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
BackendLLVMWide::llvm_load_arg (const Symbol& sym, bool derivs)
{
    ASSERT (sym.typespec().is_floatbased());
    if (sym.typespec().is_int() ||
        (sym.typespec().is_float() && !derivs)) {
        // Scalar case
    	bool is_uniform = isSymbolUniform(sym);

    	// If we are not uniform, then the argument should
    	// get passed as a pointer intstead of by value
    	// So let this case fall through
    	// NOTE:  Unclear of behavior if symbol is a constant
    	if (is_uniform) {
    		return llvm_load_value (sym, is_uniform);
    	}
    }

    if (derivs && !sym.has_derivs()) {
        // Manufacture-derivs case
        const TypeSpec &t = sym.typespec();
		// TODO:  switching back to non-wide to figure out uniform vs. varying data
        //bool temp_is_uniform = false;
    	bool temp_is_uniform = isSymbolUniform(sym);
        // Copy the non-deriv values component by component
        llvm::Value *tmpptr = llvm_alloca (t, true, temp_is_uniform);
        for (int c = 0;  c < t.aggregate();  ++c) {
            llvm::Value *v = llvm_load_value (sym, 0, c);
            llvm_store_value (v, tmpptr, t, 0, NULL, c);
        }
        // Zero out the deriv values
		// TODO:  switching back to non-wide to figure out uniform vs. varying data
        llvm::Value *zero;
        if (temp_is_uniform)
            zero = ll.constant (0.0f);
        else
        	zero = ll.wide_constant (0.0f);
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
BackendLLVMWide::llvm_store_value (llvm::Value* new_val, const Symbol& sym,
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
BackendLLVMWide::llvm_store_value (llvm::Value* new_val, llvm::Value* dst_ptr,
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

#if 1
    // TODO:  This check adds overhead, choose to remove (or not) later
    if(ll.type_ptr(ll.llvm_typeof(new_val)) != ll.llvm_typeof(dst_ptr))
    {
    	std::cerr << " new_val type=";
    	assert(0);
    	ll.llvm_typeof(new_val)->dump();
    	std::cerr << " dest_ptr type=";
    	ll.llvm_typeof(dst_ptr)->dump();
    	std::cerr << std::endl;
    }
    ASSERT(ll.type_ptr(ll.llvm_typeof(new_val)) == ll.llvm_typeof(dst_ptr));
#endif
    
    
    // Finally, store the value.
    ll.op_store (new_val, dst_ptr);
    return true;
}



bool
BackendLLVMWide::llvm_store_component_value (llvm::Value* new_val,
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
BackendLLVMWide::groupdata_field_ref (int fieldnum)
{
    return ll.GEP (groupdata_ptr(), 0, fieldnum);
}


llvm::Value *
BackendLLVMWide::groupdata_field_ptr (int fieldnum, TypeDesc type, bool is_uniform)
{
    llvm::Value *result = ll.void_ptr (groupdata_field_ref (fieldnum));
    if (type != TypeDesc::UNKNOWN) {
		if (is_uniform) {
			result = ll.ptr_to_cast (result, llvm_type(type));
		} else {
			result = ll.ptr_to_cast (result, llvm_wide_type(type));
		}
    }
    return result;
}


llvm::Value *
BackendLLVMWide::layer_run_ref (int layer)
{
    int fieldnum = 0; // field 0 is the layer_run array
    llvm::Value *layer_run = groupdata_field_ref (fieldnum);
    return ll.GEP (layer_run, 0, layer);
}



llvm::Value *
BackendLLVMWide::userdata_initialized_ref (int userdata_index)
{
    int fieldnum = 1; // field 1 is the userdata_initialized array
    llvm::Value *userdata_initiazlied = groupdata_field_ref (fieldnum);
    return ll.GEP (userdata_initiazlied, 0, userdata_index);
}



llvm::Value *
BackendLLVMWide::llvm_call_function (const char *name, 
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
    std::cout << "call_function " << name << std::endl;
    return ll.call_function (name, (valargs.size())? &valargs[0]: NULL,
                             (int)valargs.size());
}



llvm::Value *
BackendLLVMWide::llvm_call_function (const char *name, const Symbol &A,
                                 bool deriv_ptrs)
{
    const Symbol *args[1];
    args[0] = &A;
    return llvm_call_function (name, args, 1, deriv_ptrs);
}



llvm::Value *
BackendLLVMWide::llvm_call_function (const char *name, const Symbol &A,
                                 const Symbol &B, bool deriv_ptrs)
{
    const Symbol *args[2];
    args[0] = &A;
    args[1] = &B;
    return llvm_call_function (name, args, 2, deriv_ptrs);
}



llvm::Value *
BackendLLVMWide::llvm_call_function (const char *name, const Symbol &A,
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
BackendLLVMWide::llvm_test_nonzero (Symbol &val, bool test_derivs)
{
    const TypeSpec &ts (val.typespec());
    ASSERT (! ts.is_array() && ! ts.is_closure() && ! ts.is_string());
    TypeDesc t = ts.simpletype();

    // Handle int case -- guaranteed no derivs, no multi-component
    if (t == TypeDesc::TypeInt)
		// TODO:  switching back to non-wide to figure out uniform vs. varying data
        //return ll.op_ne (llvm_load_value(val), ll.wide_constant(0));
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
BackendLLVMWide::llvm_assign_impl (Symbol &Result, Symbol &Src,
                                    int arrayindex)
{
    ASSERT (! Result.typespec().is_structure());
    ASSERT (! Src.typespec().is_structure());

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

	bool is_uniform = isSymbolUniform(Result);
    // The following code handles f=f, f=i, v=v, v=f, v=i, m=m, s=s.
    // Remember that llvm_load_value will automatically convert scalar->triple.
    TypeDesc rt = Result.typespec().simpletype();
    TypeDesc basetype = TypeDesc::BASETYPE(rt.basetype);
    int num_components = rt.aggregate;
    for (int i = 0; i < num_components; ++i) {
    	llvm::Value* src_val ;
    	// Automatically handle widening the source value to match the destination's
		src_val = Src.is_constant()
			? llvm_load_constant_value (Src, arrayindex, i, basetype, is_uniform)
			: llvm_load_value (Src, 0, arrind, i, basetype, is_uniform);
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
