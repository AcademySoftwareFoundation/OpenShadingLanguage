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

#include <vector>
#include <map>

#include "oslexec_pvt.h"
#include "oslops.h"
using namespace OSL;
using namespace OSL::pvt;



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {   // OSL::pvt



/// Container for state that needs to be passed around
class RuntimeOptimizer {
public:
    RuntimeOptimizer (ShadingSystemImpl &shadingsys, ShaderGroup &group)
        : m_shadingsys(shadingsys), m_group(group),
          m_inst(NULL), m_next_newconst(0)
    {
    }

    void optimize_group ();

    /// Optimize one layer of a group, given what we know about its
    /// instance variables and connections.
    void optimize_instance ();

    /// Post-optimization cleanup of a layer: add 'useparam' instructions,
    /// track variable lifetimes, coalesce temporaries.
    void post_optimize_instance ();

    /// Set which instance we are currently optimizing.
    ///
    void set_inst (int layer);

    ShaderInstance *inst () { return m_inst; }

    ShaderGroup &group () { return m_group; }

    /// Search the instance for a constant whose type and value match
    /// type and data[...].  Return -1 if no matching const is found.
    int find_constant (const TypeSpec &type, const void *data);

    /// Search for a constant whose type and value match type and data[...],
    /// returning its index if one exists, or else creating a new constant
    /// and returning its index.  If copy is true, allocate new space and
    /// copy the data if no matching constant was found.
    int add_constant (const TypeSpec &type, const void *data);

    /// Turn the op into a simple assignment of the new symbol index to the
    /// previous first argument of the op.  That is, changes "OP arg0 arg1..."
    /// into "assign arg0 newarg".
    void turn_into_assign (Opcode &op, int newarg);

    /// Turn the op into a simple assignment of zero to the previous
    /// first argument of the op.  That is, changes "OP arg0 arg1 ..."
    /// into "assign arg0 zero".
    void turn_into_assign_zero (Opcode &op);

    /// Turn the op into a no-op.
    ///
    void turn_into_nop (Opcode &op);

    void find_constant_params (ShaderGroup &group);

    bool coerce_assigned_constant (Opcode &op);

    void make_param_use_instanceval (Symbol *R);

    /// Return the index of the symbol ultimately de-aliases to (it may be
    /// itself, if it doesn't alias to anything else).  Local block aliases
    /// are considered higher precedent than global aliases.
    int dealias_symbol (int symindex);

    /// Return the index of the symbol that 'symindex' aliases to, locally,
    /// or -1 if it has no block-local alias.
    int block_alias (int symindex) const { return m_block_aliases[symindex]; }

    /// Set the new block-local alias of 'symindex' to 'alias'.
    ///
    void block_alias (int symindex, int alias) {
        m_block_aliases[symindex] = alias;
    }

    /// Reset the block-local alias of 'symindex' so it doesn't alias to
    /// anything.
    void block_unalias (int symindex) {
        m_block_aliases[symindex] = -1;
    }

    /// Reset all block-local aliases (done when we enter a new basic
    /// block).
    void clear_block_aliases () {
        m_block_aliases.clear ();
        m_block_aliases.resize (inst()->symbols().size(), -1);
    }

    /// Set the new global alias of 'symindex' to 'alias'.
    ///
    void global_alias (int symindex, int alias) {
        m_symbol_aliases[symindex] = alias;
    }

    /// Replace R's instance value with new data.
    ///
    void replace_param_value (Symbol *R, const void *newdata);

    bool outparam_assign_elision (int opnum, Opcode &op,
                                  std::vector<bool> &in_conditional);

    bool useless_op_elision (Opcode &op);

    void make_symbol_room (int howmany=1);

    /// Add a 'useparam' before any op that reads parameters.  This is what
    /// tells the runtime that it needs to run the layer it came from, if
    /// not already done.
    void add_useparam (SymbolPtrVec &allsyms);

    void coalesce_temporaries ();

    /// Track variable lifetimes for all the symbols of the instance.
    ///
    void track_variable_lifetimes ();
    void track_variable_lifetimes (const SymbolPtrVec &allsymptrs);

    /// For each symbol, have a list of the symbols it depends on (or that
    /// depends on it).
    typedef std::map<int, std::set<int> > SymDependency;

    void syms_used_in_op (Opcode &op,
                          std::vector<int> &rsyms, std::vector<int> &wsyms);

    void track_variable_dependencies ();

    void add_dependency (SymDependency &dmap, int A, int B);

    /// Squeeze out unused symbols from an instance that has been
    /// optimized.
    void collapse_syms ();

    /// Squeeze out nop instructions from an instance that has been
    /// optimized.
    void collapse_ops ();

    /// Let the optimizer know that this (known) message was set.
    ///
    void register_message (ustring name);

    /// Let the optimizer know that an unknown message was set.
    ///
    void register_unknown_message ();

    /// Is it possible that the message with the given name was set?
    ///
    bool message_possibly_set (ustring name) const;

private:
    ShadingSystemImpl &m_shadingsys;
    ShaderGroup &m_group;             ///< Group we're optimizing
    int m_layer;                      ///< Layer we're optimizing
    ShaderInstance *m_inst;           ///< Instance we're optimizing

    // All below is just for the one inst we're optimizing:
    std::vector<int> m_all_consts;    ///< All const symbol indices for inst
    int m_next_newconst;              ///< Unique ID for next new const we add
    std::map<int,int> m_symbol_aliases; ///< Global symbol aliases
    std::vector<int> m_block_aliases;   ///< Local block aliases
    int m_local_unknown_message_sent;   ///< Non-const setmessage in this inst
    std::vector<ustring> m_local_messages_sent; ///< Messages set in this inst

    // Persistant data shared between layers
    bool m_unknown_message_sent;      ///< Somebody did a non-const setmessage
    std::vector<ustring> m_messages_sent;  ///< Names of messages set
};




/// Macro that defines the arguments to constant-folding routines
///
#define FOLDARGSDECL     RuntimeOptimizer &rop, int opnum

/// Function pointer to a constant-folding routine
///
typedef int (*OpFolder) (FOLDARGSDECL);

/// Macro that defines the full declaration of a shadeop constant-folder.
/// 
#define DECLFOLDER(name)  int name (FOLDARGSDECL)




}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
