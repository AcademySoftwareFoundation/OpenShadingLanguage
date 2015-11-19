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

#pragma once

#include <vector>
#include <map>

#include <boost/version.hpp>
#if BOOST_VERSION >= 104900
# include <boost/container/flat_map.hpp>
# define USE_FLAT_MAP 1
#endif

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;


OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt

#if USE_FLAT_MAP
typedef boost::container::flat_map<int,int> FastIntMap;
#else
typedef std::map<int,int> FastIntMap;
#endif



/// OSOProcessor that does runtime optimization on shaders.
class RuntimeOptimizer : public OSOProcessorBase {
public:
    RuntimeOptimizer (ShadingSystemImpl &shadingsys, ShaderGroup &group,
                      ShadingContext *context);

    virtual ~RuntimeOptimizer ();

    virtual void run ();

    virtual void set_inst (int layer);

    virtual void set_debug ();

    /// Optimize one layer of a group, given what we know about its
    /// instance variables and connections.
    void optimize_instance ();

    /// One optimization pass over a range of instructions [begin, end).
    /// Return the number of changes made.
    int optimize_ops (int beginop, int endop);

    /// Post-optimization cleanup of a layer: add 'useparam' instructions,
    /// track variable lifetimes, coalesce temporaries.
    void post_optimize_instance ();

    /// What's our current optimization level?
    int optimize() const { return m_optimize; }

    /// Search the instance for a constant whose type and value match
    /// type and data[...].  Return -1 if no matching const is found.
    int find_constant (const TypeSpec &type, const void *data);

    /// Search for a constant whose type and value match type and data[...],
    /// returning its index if one exists, or else creating a new constant
    /// and returning its index.
    int add_constant (const TypeSpec &type, const void *data,
                      TypeDesc datatype=TypeDesc::UNKNOWN);
    int add_constant (float c) { return add_constant(TypeDesc::TypeFloat, &c); }
    int add_constant (int c) { return add_constant(TypeDesc::TypeInt, &c); }
    int add_constant (ustring s) { return add_constant(TypeDesc::TypeString, &s); }

    /// Create a new temporary variable of the given type, return its index.
    int add_temp (const TypeSpec &type);

    /// Search for the given global, adding it to the symbol table if
    /// necessary, and returning its index.
    int add_global (ustring name, const TypeSpec &type);

    /// Add a new symbol to the current instance's symbol list. Don't push
    /// onto the symbol table yourself during optimization; this does some
    /// other essential housekeeping.
    int add_symbol (const Symbol &sym);

    /// Turn the op into a simple assignment of the new symbol index to the
    /// previous first argument of the op.  That is, changes "OP arg0 arg1..."
    /// into "assign arg0 newarg".
    void turn_into_assign (Opcode &op, int newarg, string_view why=NULL);

    /// Turn the op into a simple assignment of zero to the previous
    /// first argument of the op.  That is, changes "OP arg0 arg1 ..."
    /// into "assign arg0 zero".
    void turn_into_assign_zero (Opcode &op, string_view why=NULL);

    /// Turn the op into a simple assignment of one to the previous
    /// first argument of the op.  That is, changes "OP arg0 arg1 ..."
    /// into "assign arg0 one".
    void turn_into_assign_one (Opcode &op, string_view why=NULL);

    /// Turn the op into a new simple unary or binary op with arguments
    /// newarg0 (the result, newarg1, and newarg2.  If newarg2 < 0, then
    /// it's a unary op, otherwise a binary op.  The original arg list must
    /// have at least as many operands as the new one, since no new arg
    /// space is allocated.
    void turn_into_new_op (Opcode &op, ustring newop, int newarg0,
                           int newarg1, int newarg2, string_view why=NULL);

    /// Turn the op into a no-op.  Return 1 if it changed, 0 if it was
    /// already a nop.
    int turn_into_nop (Opcode &op, string_view why=NULL);

    /// Turn the whole range [begin,end) into no-ops.  Return the number
    /// of instructions that were altered.
    int turn_into_nop (int begin, int end, string_view why=NULL);

    void debug_opt_impl (string_view message) const;
    TINYFORMAT_WRAP_FORMAT (void, debug_opt, const,
                            std::ostringstream msg;, msg,
                            debug_opt_impl(msg.str());)
    void debug_opt_ops (int opbegin, int opend, string_view message) const;
    void debug_turn_into (const Opcode &op, int numops,
                          string_view newop, int newarg0,
                          int newarg1, int newarg2, string_view why);

    void simplify_params ();

    void find_params_holding_globals ();

    bool coerce_assigned_constant (Opcode &op);

    void make_param_use_instanceval (Symbol *R, string_view why=NULL);

    /// Return the index of the symbol ultimately de-aliases to (it may be
    /// itself, if it doesn't alias to anything else).  Local block aliases
    /// are considered higher precedent than global aliases.
    int dealias_symbol (int symindex, int opnum=-1);

    /// Return the index of the symbol that 'symindex' aliases to, locally,
    /// or -1 if it has no block-local alias.
    int block_alias (int symindex) const {
        FastIntMap::const_iterator i = m_block_aliases.find (symindex);
        return (i == m_block_aliases.end()) ? -1 : i->second;
    }

    /// Set the new block-local alias of 'symindex' to 'alias'.
    ///
    void block_alias (int symindex, int alias) {
        m_block_aliases[symindex] = alias;
    }

    /// Reset the block-local alias of 'symindex' so it doesn't alias to
    /// anything.
    void block_unalias (int symindex) {
        FastIntMap::iterator i = m_block_aliases.find (symindex);
        if (i != m_block_aliases.end())
            i->second = -1;
    }

    /// Clear local block aliases for any args that are written by this op.
    void block_unalias_written_args (Opcode &op) {
        for (int i = 0, e = op.nargs();  i < e;  ++i)
            if (op.argwrite(i))
                block_unalias (inst()->arg(op.firstarg()+i));
    }

    /// Reset all block-local aliases (done when we enter a new basic
    /// block).
    void clear_block_aliases () {
        m_block_aliases.clear ();
    }

    /// Set the new global alias of 'symindex' to 'alias'.
    ///
    void global_alias (int symindex, int alias) {
        m_symbol_aliases[symindex] = alias;
    }

    /// Is the given symbol stale?  A "stale" symbol is one that, within
    /// the current basic block, has been assigned in a simple manner
    /// (by a single op with no other side effects), but not yet used.
    /// The point is that if they are simply assigned again before being
    /// used, that first assignment can be turned into a no-op.
    bool sym_is_stale (int sym) {
        return m_stale_syms.find(sym) != m_stale_syms.end();
    }

    /// Clear the stale symbol list -- we do this when entering a new
    /// basic block.
    void clear_stale_syms ();

    /// Take a symbol out of the stale list -- we do this when a symbol
    /// is used in any way.
    void use_stale_sym (int sym);

    /// Is the op a "simple" assignment (arg 0 completely overwritten,
    /// no side effects or funny business)?
    bool is_simple_assign (Opcode &op);

    /// Called when symbol sym is "simply" assigned at the given op.  An
    /// assignment is considered simple if it completely overwrites the
    /// symbol in a single op and has no side effects.  When this
    /// happens, we mark the symbol as "stale", meaning it's got a value
    /// that hasn't been read yet.  If it's wholy assigned again before
    /// it's read, we can go back and remove the earlier assignment.
    void simple_sym_assign (int sym, int op);

    /// Return true if assignments to A on this op have no effect because
    /// they will not be subsequently used.
    bool unread_after (const Symbol *A, int opnum);

    /// Replace R's instance value with new data.
    ///
    void replace_param_value (Symbol *R, const void *newdata,
                              const TypeSpec &newdata_type);

    bool outparam_assign_elision (int opnum, Opcode &op);

    bool useless_op_elision (Opcode &op, int opnum);

    void make_symbol_room (int howmany=1);

    enum RecomputeRWRangesOption { DontRecomputeRWRanges, RecomputeRWRanges };
    enum InsertRelation { NoRelation=0, GroupWithPrevious=-1, GroupWithNext=1 };
    /// Insert instruction 'opname' with arguments 'args_to_add' into
    /// the code at instruction 'opnum'.  The existing code and
    /// concatenated argument lists can be found in code and opargs,
    /// respectively, and allsyms contains pointers to all symbols.
    /// mainstart is a reference to the address where the 'main' shader
    /// begins, and may be modified if the new instruction is inserted
    /// before that point.  The recompute_rw_ranges parameter determines
    /// whether all symbols' read/write ranges should be adjusted to
    /// take the new instruction into consideration.  Relation indicates
    /// its relation to surrounding instructions: GroupWithPrevious
    /// means it should have the same method, sourcefile, and sourceline
    /// as the preceeding instruction; GroupWithNext means it should
    /// have the same method, sourcefile, and sourceline as the
    /// subsequent instruction; NoRelation means we have no information,
    /// so don't copy that info from anywhere.
    void insert_code (int opnum, ustring opname,
                      const std::vector<int> &args_to_add,
                      RecomputeRWRangesOption recompute_rw_ranges,
                      InsertRelation relation=GroupWithNext);
    /// insert_code with begin/end arg array pointers.
    void insert_code (int opnum, ustring opname,
                      const int *argsbegin, const int *argsend,
                      RecomputeRWRangesOption recompute_rw_ranges,
                      InsertRelation relation=GroupWithNext);
    /// insert_code with explicit arguments (up to 4, a value of -1 means
    /// the arg isn't used).  Presume recompute_rw_ranges is true.
    void insert_code (int opnum, ustring opname, InsertRelation relation,
                      int arg0=-1, int arg1=-1, int arg2=-1, int arg3=-1);

    void insert_useparam (size_t opnum, const std::vector<int> &params_to_use);

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
    typedef std::set<int> SymIntSet;
    typedef std::map<int, SymIntSet> SymDependency;

    void syms_used_in_op (Opcode &op,
                          std::vector<int> &rsyms, std::vector<int> &wsyms);

    void track_variable_dependencies ();

    void add_dependency (SymDependency &dmap, int A, int B);

    void mark_symbol_derivatives (SymDependency &symdeps, SymIntSet &visited, int d);

    void mark_outgoing_connections ();

    int remove_unused_params ();

    /// Turn isconnected() calls into constant assignments
    void resolve_isconnected ();

    int eliminate_middleman ();

    /// Squeeze out unused symbols from an instance that has been
    /// optimized.
    void collapse_syms ();

    /// Squeeze out nop instructions from an instance that has been
    /// optimized.
    void collapse_ops ();

    /// Let the optimizer know that this (known, constant) message was
    /// set by the current instance.
    void register_message (ustring name);

    /// Let the optimizer know that an unknown message (i.e., we
    /// couldn't reduce the message name to a constant) was set by the
    /// current instance.
    void register_unknown_message ();

    /// Is it possible that the message with the given name was set?
    ///
    bool message_possibly_set (ustring name) const;

    /// Return the index of the next instruction within the same basic
    /// block that isn't a NOP.  If there are no more non-NOP
    /// instructions in the same basic block as opnum, return 0.
    int next_block_instruction (int opnum);

    /// Perform peephole optimization on pairs of adjacent instructions
    /// within the same basic block.
    int peephole2 (int opnum, int op2num);

    /// Perform various optimizations specific to assignments.
    int optimize_assignment (Opcode &op, int opnum);

    bool opt_elide_unconnected_outputs () const {
        return m_opt_elide_unconnected_outputs;
    }

    /// Are special optimizations to 'mix' requested?
    bool opt_mix () const { return m_opt_mix; }

    /// Which optimization pass are we on?
    int optimization_pass () const { return m_pass; }

    /// Retrieve ptr to the dummy shader globals
    ShaderGlobals *shaderglobals () { return &m_shaderglobals; }

    // Maximum number of new constant symbols that a constant-folding
    // function is able to add.
    static const int max_new_consts_per_fold = 10;

    void stop_optimizing () { m_stop_optimizing = true; }

    std::string op_string (const Opcode &op) {
        std::string s = op.opname().string();
        for (int a = 0;  a < op.nargs();  ++a)
            s = s + ' ' + opargsym(op,a)->name().string();
        return s;
    }

    std::ostream & printinst (std::ostream &out) const;

private:
    int m_optimize;                   ///< Current optimization level
    bool m_opt_simplify_param;            ///< Turn instance params into const?
    bool m_opt_constant_fold;             ///< Allow constant folding?
    bool m_opt_stale_assign;              ///< Optimize stale assignments?
    bool m_opt_elide_useless_ops;         ///< Optimize away useless ops?
    bool m_opt_elide_unconnected_outputs; ///< Optimize unconnected outputs?
    bool m_opt_peephole;                  ///< Do some peephole optimizations?
    bool m_opt_coalesce_temps;            ///< Coalesce temporary variables?
    bool m_opt_assign;                    ///< Do various assign optimizations?
    bool m_opt_mix;                       ///< Do mix optimizations?
    bool m_opt_middleman;                 ///< Do middleman optimizations?
    ShaderGlobals m_shaderglobals;        ///< Dummy ShaderGlobals

    // Keep track of some things for the whole shader group:
    typedef boost::unordered_map<ustring,ustring,ustringHash> ustringmap_t;
    std::vector<ustringmap_t> m_params_holding_globals;
                   ///< Which params of each layer really just hold globals

    // All below is just for the one inst we're optimizing at the moment:
    int m_pass;                       ///< Optimization pass we're on now
    std::vector<int> m_all_consts;    ///< All const symbol indices for inst
    int m_next_newconst;              ///< Unique ID for next new const we add
    int m_next_newtemp;               ///< Unique ID for next new temp we add
    FastIntMap m_symbol_aliases;      ///< Global symbol aliases
    FastIntMap m_block_aliases;         ///< Local block aliases
    FastIntMap m_param_aliases;         ///< Params aliasing to params/globals
    FastIntMap m_stale_syms;            ///< Stale symbols for this block
    int m_local_unknown_message_sent;   ///< Non-const setmessage in this inst
    std::vector<ustring> m_local_messages_sent; ///< Messages set in this inst
    std::set<ustring> m_textures_needed;
    std::set<ustring> m_closures_needed;
    std::set<ustring> m_globals_needed;
    std::set<AttributeNeeded> m_attributes_needed;
    bool m_unknown_textures_needed;
    bool m_unknown_closures_needed;
    bool m_unknown_attributes_needed;
    std::set<UserDataNeeded> m_userdata_needed;
    double m_stat_opt_locking_time;       ///<   locking time
    double m_stat_specialization_time;    ///<   specialization time
    bool m_stop_optimizing;           ///< for debugging

    // Persistant data shared between layers
    bool m_unknown_message_sent;      ///< Somebody did a non-const setmessage
    std::vector<ustring> m_messages_sent;  ///< Names of messages set

    friend class ShadingSystemImpl;
};



/// Macro that defines the arguments to constant-folding routines
///
#define FOLDARGSDECL     RuntimeOptimizer &rop, int opnum

/// Macro that defines the full declaration of a shadeop constant-folder.
/// 
#define DECLFOLDER(name)  int name (FOLDARGSDECL)



}; // namespace pvt
OSL_NAMESPACE_EXIT
