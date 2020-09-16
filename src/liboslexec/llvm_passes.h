// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <utility>

#include <OSL/oslconfig.h>

#include <llvm/Pass.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>

OSL_NAMESPACE_ENTER

namespace pvt {

namespace {

// When a platform doesn't have a native data type that represents a LLVM
// bit mask <32xi1>, <16xi1>, <8xi1>, or <4xi1>, the instruction lowering
// will select the smallest type that can represent the bitmask.  IE: for
// AVX & AVX2, which is <8xi1> will be promoted to <8xi16>.  This is
// unfortunate as it will then generate 6 additional instructions to convert
// <8xi16> back to <8xi32> for use in 32 bit vector operations.  However it
// would have been correct for 16 bit operations. Thus the whole point of
// this optimization pass is to avoid that situation by preventing LLVM bit
// masks (<32xi1>, <16xi1>, <8xi1>, or <4xi1>) from passing between basic
// blocks (becoming 'liveins' for another basic block). This is somewhat an
// artifact of LLVM instruction lowering happening at each basic block vs. a
// higher level (function or globally).  Should future LLVM versions change
// how instruction lowering happens then this pass may not be necessary.
// Also if future LLVM version takes on the work of this optimization pass,
// then it may be removed.
template <int WidthT>
class PreventBitMasksFromBeingLiveinsToBasicBlocks
    : public llvm::FunctionPass
{
    typedef llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter> IRBuilder;
    llvm::Type* m_llvm_mask_type;
    llvm::Type* m_native_mask_type;
    llvm::Constant* m_wide_zero_initializer;
    std::unordered_map<llvm::Instruction*, llvm::Value*> m_native_mask_by_producing_inst;
    std::unordered_map<llvm::Value*, llvm::Value*> m_llvm_mask_from_livein_by_native_mask;
    std::vector<llvm::Instruction*> m_phiNodesWithNativeMasks;

public:
    static char ID;

    PreventBitMasksFromBeingLiveinsToBasicBlocks()
        : FunctionPass(ID)
        , m_llvm_mask_type(nullptr)
        , m_native_mask_type(nullptr)
        , m_wide_zero_initializer(nullptr)
    {}

    bool doInitialization (llvm::Module &M) override {
        llvm::Type* llvm_type_bool = llvm::Type::getInt1Ty (M.getContext());
        llvm::Type* llvm_type_int32 = llvm::Type::getInt32Ty (M.getContext());

        m_llvm_mask_type = llvm::VectorType::get(llvm_type_bool, WidthT);

        // NOTE:  OSL doesn't have any 16 bit data types, so 32bit version
        // of the mask promotion will always be correct here.  Should 16 bit
        // support be needed, this pass could be extended to look at the
        // other operands of the select and other instructions to decide
        // what the native mask type should be.  And if necessary maintain a
        // 16 bit and 32 bit native mask representation to be passed as a
        // livein.
        m_native_mask_type = llvm::VectorType::get(llvm_type_int32, WidthT);

        m_wide_zero_initializer = llvm::ConstantVector::getSplat(WidthT, llvm::ConstantInt::get (M.getContext(), llvm::APInt(32,0)));

        return false; // I don't think we modified the module
    }

    bool runOnFunction(llvm::Function &F) override {
        OSL_DEV_ONLY(llvm::errs() << ">>>>>>>>>>>>>>>>>>PreventBitMasksFromBeingLiveinsToBasicBlocks<" << WidthT << ">:");
        OSL_DEV_ONLY(llvm::errs().write_escaped(F.getName()) << '\n');

        m_native_mask_by_producing_inst.clear();

        bool changed = false;

        for (llvm::BasicBlock &bb : F) {
            OSL_DEV_ONLY(llvm::errs() << ">>>>>>>>>Basic Block: ");
            OSL_DEV_ONLY(llvm::errs().write_escaped(bb.getName()) << '\n');

            m_phiNodesWithNativeMasks.clear();
            m_llvm_mask_from_livein_by_native_mask.clear();

            for (llvm::Instruction &inst : bb) {
                // We could possibly identify all the instruction types that
                // we "think" could be using a mask, but feel just looking
                // at the operand types is cheaper.

                // Although we do have special case phi nodes
                bool isPhi = llvm::dyn_cast<llvm::PHINode>(&inst) != nullptr;

                if (!isPhi && !m_phiNodesWithNativeMasks.empty()) {
                    // As all phi nodes have to appear at the top of a basic
                    // block, if an instruction is not a phi, then we are
                    // past the block of phi nodes and this is a good
                    // location to convert the phi nodes with native masks
                    // back to llvm masks.
                    IRBuilder builder(&bb);
                    builder.SetInsertPoint(&inst);
                    for (llvm::Instruction* phiNodeWithNativeMask : m_phiNodesWithNativeMasks) {
                        llvm::Value* llvm_mask = builder.CreateICmpSLT (phiNodeWithNativeMask,m_wide_zero_initializer);
                        // Now we need to replace all uses of phiNodeWithNativeMask with our converted llvm mask
                        auto use_iter = phiNodeWithNativeMask->use_begin(),
                                           use_end = phiNodeWithNativeMask->use_end();
                        for (; use_iter != use_end;)
                        {
                            llvm::Use &use = *use_iter;
                            ++use_iter;
                            auto *user = llvm::dyn_cast<llvm::Instruction>(use.getUser());
                            // We need to skip the use wher the user is our conversion to a llvm_mask
                            if (user && user == llvm_mask)
                                continue;
                            use.set(llvm_mask);
                       }
                    }

                    m_phiNodesWithNativeMasks.clear();

                    // NOTE: we did add instructions to this bb, but the
                    // should be inserted before the current inst. Assume
                    // underlying implementation of the instruction list is
                    // truly a linked list of some kind, inserting before
                    // the current iterator should not invalidate the
                    // iterator.
                }

                bool phiNodeHadOperandReplaced = false;
                bool phiNodeHadOperandReplacedInLastPassOverOperands;

                OSL_DEV_ONLY(llvm::errs() << ">>>>Op: ");
                OSL_DEV_ONLY(llvm::errs().write_escaped(inst.getOpcodeName()) << '\n');

                // As we may have multiple operands that need to be replaced
                // for a phi node, once we discover a qualifying replacement
                // we will need to reprocess the other operands. We choose
                // to just repeat the original algorithm over the operands
                // until no additional phi node replacements occur. Although
                // technically it should only be 2 passes, 1 pass to
                // discover 1 or more operands need to be replaced and a 2nd
                // pass to fix up any other mask types that might need it
                // (mask from same BB or constants).
                do {
                    phiNodeHadOperandReplacedInLastPassOverOperands = false;

                    // As we could be replacing operands, but not the number of operands
                    // we use index vs. iterator based loop construct
                    for (unsigned operand_i = 0, num_operands = inst.getNumOperands();
                         operand_i != num_operands;
                         ++operand_i) {

                        llvm::Value *op_val = inst.getOperand (operand_i);
                        if (op_val != nullptr) {
                            llvm::Type* op_type =  op_val->getType();
                            if (op_type == m_llvm_mask_type) {
                                // See if the value came from an instruction
                                auto *producing_instr = llvm::dyn_cast<llvm::Instruction>(op_val);
                                if (producing_instr != nullptr) {
                                    llvm::BasicBlock* producing_bb = producing_instr->getParent();

                                    // If a phi node needs to have one operand promoted,
                                    // it will need all operands promoted even if they were
                                    // produced in the same basic block, so the types match
                                    if (phiNodeHadOperandReplaced || producing_bb != &bb) {
                                        OSL_DEV_ONLY(llvm::errs() << "Uses llvm mask <" <<WidthT << "xi1> from different Basic Block: ");
                                        OSL_DEV_ONLY(llvm::errs().write_escaped(op_val->getName()) << '\n');
                                        changed = true;

                                        // We can't let an unrepresentable data type of <16xi1>, <8xi1> or <4xi1>
                                        // flow between basic blocks
                                        // So we will sign extend the bitmask to <16xi32>, <8xi32> or <4xi32>
                                        // inside the basic block that produced the mask
                                        llvm::Value* native_mask = nullptr;
                                        {
                                            // We may have already created a native mask for this instruction
                                            auto search_result = m_native_mask_by_producing_inst.find(producing_instr);
                                            if (search_result == m_native_mask_by_producing_inst.end()) {

                                                // Scan producing basic block for an existing sign extend instruction
                                                // for the producing_instr.  As the existence of a basic block means
                                                // we most likely have a branch higher up whose test required a sign extend
                                                for (llvm::Instruction &other_inst : *producing_bb) {
                                                    auto *existing_sign_ext = llvm::dyn_cast<llvm::SExtInst>(&other_inst);
                                                    if (existing_sign_ext) {
                                                        OSL_ASSERT(existing_sign_ext->getNumOperands() == 1);
                                                        llvm::Value *existing_operand = existing_sign_ext->getOperand (0);
                                                        if (existing_operand == producing_instr) {
                                                            OSL_ASSERT(existing_sign_ext->getType() == m_native_mask_type);
                                                            OSL_DEV_ONLY(llvm::errs() << "Using existing sign ext in producing bb\n");

                                                            native_mask = existing_sign_ext;
                                                            break;
                                                        }
                                                    }
                                                }
                                                if (nullptr == native_mask) {
                                                    IRBuilder builder(producing_bb);
                                                    builder.SetInsertPoint(&producing_bb->back());

                                                    native_mask = builder.CreateSExt(producing_instr, m_native_mask_type);
                                                    // NOTE: we did add instructions, but not to the BB we are currently
                                                    // iterating over instructions
                                                }
                                                m_native_mask_by_producing_inst.insert(std::make_pair(producing_instr,native_mask));
                                            } else {
                                                native_mask = search_result->second;
                                            }
                                        }
                                        OSL_ASSERT(native_mask);

                                        if (!isPhi) {
                                            // Then inside the current basic block convert the <16xi32>, <8xi32> or <4xi32>
                                            // back to llvm's bit mask of <16xi1>, <8xi1> or <4xi1>
                                            // Ultimately we expect instruction selection to replace the <8xi1> or <4xi1>
                                            // with compatible <8xi32> or <4xi32> which should have the net result of
                                            // eliminating all these conversions.
                                            llvm::Value* llvm_mask = nullptr;
                                            {
                                                // We may have already created a llvm mask in this basic block for the native mask
                                                auto search_result = m_llvm_mask_from_livein_by_native_mask.find(native_mask);
                                                if (search_result == m_llvm_mask_from_livein_by_native_mask.end()) {
                                                    // Insert the conversion from native to llvm mask
                                                    // somewhere before the 1st instruction that needs to use it in this basic block
                                                    IRBuilder builder(&bb);
                                                    builder.SetInsertPoint(&inst);

                                                    llvm_mask = builder.CreateICmpSLT (native_mask,m_wide_zero_initializer);

                                                    m_llvm_mask_from_livein_by_native_mask.insert(std::make_pair(native_mask,llvm_mask));

                                                    // NOTE: we did add instructions to this bb,
                                                    // but the should be inserted before the current inst.
                                                    // Assume underlying implementation of the instruction list is
                                                    // truly a linked list of some kind, inserting before the current
                                                    // iterator should not invalidate the iterator

                                                } else {
                                                    llvm_mask = search_result->second;
                                                }
                                            }
                                            inst.setOperand (operand_i, llvm_mask);
                                        } else {

                                            // Phi nodes have to exist at the top of a basic block which precludes
                                            // our basic algorithm from just converting the native mask back to a llvm mask
                                            // before the phi node.
                                            // Instead we will need to let the Phi node operate on the native mask types
                                            // and insert an instruction to convert it after the block of phi nodes
                                            // However, this now means we will need to find all uses of the phi node's
                                            // result and replace them with our converted llvm_mask
                                            inst.setOperand (operand_i, native_mask);
                                            phiNodeHadOperandReplaced = true;
                                            phiNodeHadOperandReplacedInLastPassOverOperands = true;
                                        }
                                        // We should be able to continue iterating over the rest of the operands
                                    }
                                } else if (phiNodeHadOperandReplaced) {
                                    auto *constant = llvm::dyn_cast<llvm::Constant>(op_val);
                                    if (constant) {
                                        OSL_DEV_ONLY(llvm::errs() << "Uses constant llvm mask <" <<WidthT << "xi1> \n");

                                        // Should handle promoting whatever the constant value is (most likely zeroinitializer)
                                        llvm::ConstantFolder Folder;
                                        auto *signExtConstant = Folder.CreateCast(llvm::Instruction::SExt, constant, m_native_mask_type);

                                        inst.setOperand (operand_i, signExtConstant);
                                    } else {
                                        OSL_ASSERT(0 && "Unhandled/Unexpected llvm mask type");
                                    }
                                }
                            }
                        }
                    }
                } while (phiNodeHadOperandReplacedInLastPassOverOperands);
                if (phiNodeHadOperandReplaced) {
                    // Since we changed out the operands of the phi with a
                    // native mask, we need to mutate the type of the phi
                    // itself.
                    // NOTE: docs claim this is dangerous, but logically
                    // fits with what we are doing. One possible issue would
                    // be any optimization meta data or other lookup tables
                    // built elsewhere that depend on the type of the phi
                    // instructions.
                    inst.mutateType(m_native_mask_type);

                    m_phiNodesWithNativeMasks.push_back(&inst);
                }
            }
        }
        OSL_DEV_ONLY(llvm::errs() << ">>>>>>>>>>>>>>>>>>\n\n");
        return changed;
    }
};



template <int WidthT>
class PrePromoteLogicalOpsOnBitMasks
    : public llvm::FunctionPass
{
    typedef llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter> IRBuilder;
    llvm::Type* m_llvm_mask_type;
    llvm::Type* m_native_mask_type;
    llvm::Constant* m_wide_zero_initializer;
    std::unordered_map<llvm::Instruction*, llvm::Value*> m_native_mask_by_producing_inst;

public:
    static char ID;

    PrePromoteLogicalOpsOnBitMasks()
        : FunctionPass(ID)
        , m_llvm_mask_type(nullptr)
        , m_native_mask_type(nullptr)
        , m_wide_zero_initializer(nullptr)
    {}

    bool doInitialization (llvm::Module &M) override {
        llvm::Type* llvm_type_bool = llvm::Type::getInt1Ty(M.getContext());
        llvm::Type* llvm_type_int32 = llvm::Type::getInt32Ty(M.getContext());

        m_llvm_mask_type = llvm::VectorType::get(llvm_type_bool, WidthT);

        // NOTE:  OSL doesn't have any 16 bit data types, so 32bit version
        // of the mask promotion will always be correct here.  Should 16 bit
        // support be needed, this pass could be extended.
        m_native_mask_type = llvm::VectorType::get(llvm_type_int32, WidthT);

        m_wide_zero_initializer = llvm::ConstantVector::getSplat(WidthT, llvm::ConstantInt::get (M.getContext(), llvm::APInt(32,0)));
        return false; // I don't think we modified the module
    }

    bool runOnFunction(llvm::Function &F) override {
        OSL_DEV_ONLY(llvm::errs() << ">>>>>>>>>>>>>>>>>>PrePromoteLogicalOpsOnBitMasks<" << WidthT << ">:");
        OSL_DEV_ONLY(llvm::errs().write_escaped(F.getName()) << '\n');

        m_native_mask_by_producing_inst.clear();

        bool changed = false;

        for (llvm::BasicBlock &bb : F) {
            OSL_DEV_ONLY(llvm::errs() << ">>>>>>>>>Basic Block: ");
            OSL_DEV_ONLY(llvm::errs().write_escaped(bb.getName()) << '\n');

            for (llvm::Instruction &inst : bb) {
                auto *binary_op = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
                if (binary_op) {
                    llvm::Type* logical_op_type =  binary_op->getType();
                    if (logical_op_type == m_llvm_mask_type) {
                        llvm::Instruction::BinaryOps bin_opcode = binary_op->getOpcode();
                        if (bin_opcode == llvm::Instruction::And ||
                            bin_opcode == llvm::Instruction::Or ||
                            bin_opcode == llvm::Instruction::Xor) {

                            OSL_DEV_ONLY(llvm::errs() << ">>>>Binary Logical Op: ");
                            OSL_DEV_ONLY(llvm::errs().write_escaped(binary_op->getOpcodeName()) << '\n');
                            OSL_ASSERT(inst.getNumOperands() == 2);
                            for (unsigned operand_i = 0; operand_i != 2; ++operand_i) {
                                llvm::Value *op_val = inst.getOperand (operand_i);
                                OSL_ASSERT(op_val != nullptr);
                                llvm::Type* op_type =  op_val->getType();
                                OSL_ASSERT(op_type == m_llvm_mask_type);

                                // See if the value came from an instruction
                                auto *producing_instr = llvm::dyn_cast<llvm::Instruction>(op_val);
                                if (producing_instr != nullptr) {
                                    llvm::BasicBlock* producing_bb = producing_instr->getParent();

                                    // As we have not 16bit operations, we can't let an unrepresentable
                                    // data type of <16xi1>, <8xi1> or <4xi1> be promoted to the default
                                    // <16xi16>, <8xi16> or <4xi16> data type which will then require
                                    // extra instructions to promoted it again to xi32.
                                    // So we will sign extend the bitmask to <16xi32>, <8xi32> or <4xi32>
                                    // inside the basic block that produced the mask
                                    llvm::Value* native_mask = nullptr;
                                    {
                                        // We may have already created a native mask for this instruction
                                        auto search_result = m_native_mask_by_producing_inst.find(producing_instr);
                                        if (search_result == m_native_mask_by_producing_inst.end()) {

                                            // Scan producing basic block for an existing sign extend instruction
                                            // for the producing_instr.
                                            for (llvm::Instruction &other_inst : *producing_bb) {
                                                auto *existing_sign_ext = llvm::dyn_cast<llvm::SExtInst>(&other_inst);
                                                if (existing_sign_ext) {
                                                    OSL_ASSERT(existing_sign_ext->getNumOperands() == 1);
                                                    llvm::Value *existing_operand = existing_sign_ext->getOperand (0);
                                                    if (existing_operand == producing_instr) {
                                                        OSL_ASSERT(existing_sign_ext->getType() == m_native_mask_type);
                                                        OSL_DEV_ONLY(llvm::errs() << "Using existing sign ext in producing bb\n");

                                                        if (producing_bb == &bb) {
                                                            // To reuse the existing sign ext, we have to make sure that
                                                            // instruction exists before our use of it
                                                            bool inst_is_before_sign_ext = false;
                                                            llvm::Instruction * prev_instr = existing_sign_ext;
                                                            do {
                                                                prev_instr = prev_instr->getPrevNode();
                                                                if (prev_instr == &inst) {
                                                                    inst_is_before_sign_ext = true;
                                                                    break;
                                                                }
                                                            } while (prev_instr != nullptr);
                                                            if (inst_is_before_sign_ext) {
                                                                existing_sign_ext->moveBefore(&inst);
                                                            }
                                                        }
                                                        native_mask = existing_sign_ext;
                                                        break;
                                                    }
                                                }
                                            }
                                            if (nullptr == native_mask) {
                                                IRBuilder builder(producing_bb);
                                                llvm::Instruction *insert_point = producing_instr->getNextNode();
                                                OSL_ASSERT(insert_point != nullptr);
                                                // We can't insert our conversion instruction in a middle
                                                // of a Phi block, so iterate over instructions until we
                                                // get to end of phi block
                                                while(llvm::dyn_cast<llvm::PHINode>(insert_point) != nullptr) {
                                                    insert_point = insert_point->getNextNode();
                                                    OSL_ASSERT(insert_point != nullptr);
                                                }
                                                builder.SetInsertPoint(insert_point);

                                                native_mask = builder.CreateSExt(producing_instr, m_native_mask_type);
                                                // NOTE: we did add instructions, but not to the BB we are currently
                                                // iterating over instructions
                                            }
                                            m_native_mask_by_producing_inst.insert(std::make_pair(producing_instr,native_mask));
                                        } else {
                                            native_mask = search_result->second;
                                        }
                                    }
                                    OSL_ASSERT(native_mask);
                                    inst.setOperand (operand_i, native_mask);
                                } else {
                                    auto *constant = llvm::dyn_cast<llvm::Constant>(op_val);
                                    if (constant) {
                                        OSL_DEV_ONLY(llvm::errs() << "Uses constant llvm mask <" <<WidthT << "xi1> \n");

                                        // Should handle promoting whatever the constant value is (most likely zeroinitializer)
                                        llvm::ConstantFolder Folder;
                                        auto *signExtConstant = Folder.CreateCast(llvm::Instruction::SExt, constant, m_native_mask_type);

                                        inst.setOperand (operand_i, signExtConstant);
                                    } else {
                                        OSL_ASSERT(0 && "Unhandled/Unexpected llvm mask type");
                                    }
                                }
                            }
                            changed = true;

                            // Since we changed out the operands of the logical op with native masks, we need
                            // to mutate the type of the logical op itself
                            // NOTE: docs claim this is dangerous, but logically fits with what we are doing.
                            // One possible issue would be any optimization meta data or other lookup tables
                            // built elsewhere that depend on the type of the phi instructions
                            inst.mutateType(m_native_mask_type);

                            // Then right after the logical instruction convert the result of the logical operation
                            // from <16xi32>, <8xi32> or <4xi32> back to llvm's bit mask of <16xi1>, <8xi1> or <4xi1>
                            // Ultimately we expect instruction selection to replace the <8xi1> or <4xi1>
                            // with compatible <8xi32> or <4xi32> which should have the net result of
                            // eliminating all these conversions.
                            IRBuilder builder(&bb);
                            OSL_ASSERT(inst.getNextNode() != nullptr);
                            builder.SetInsertPoint(inst.getNextNode());

                            llvm::Value* llvm_mask = builder.CreateICmpSLT (&inst,m_wide_zero_initializer);
                            // Now we need to replace all uses of phiNodeWithNativeMask with our converted llvm mask
                            auto use_iter = inst.use_begin();
                            auto use_end = inst.use_end();
                            for (; use_iter != use_end;) {
                                llvm::Use &use = *use_iter;
                                ++use_iter;
                                auto *user = llvm::dyn_cast<llvm::Instruction>(use.getUser());
                                // We need to skip the use wher the user is our conversion to a llvm_mask
                                if (user && user == llvm_mask)
                                    continue;
                                use.set(llvm_mask);
                            }
                        }
                    }
                }
            }
        }
        OSL_DEV_ONLY(llvm::errs() << ">>>>>>>>>>>>>>>>>>\n\n");
        return changed;
    }
};



// No need to worry about static variable collisions if included multiple
// places because of the anonymous namespace, each translation unit
// including this file will need its own static members defined. LLVM will
// assign IDs when they get registered, so this initialization value is not
// important.
template<>
char PreventBitMasksFromBeingLiveinsToBasicBlocks<8>::ID = 0;

template<>
char PreventBitMasksFromBeingLiveinsToBasicBlocks<16>::ID = 0;

template<>
char PrePromoteLogicalOpsOnBitMasks<8>::ID = 0;

template<>
char PrePromoteLogicalOpsOnBitMasks<16>::ID = 0;

}  // end of anonymous namespace

}  // namespace pvt
OSL_NAMESPACE_EXIT
