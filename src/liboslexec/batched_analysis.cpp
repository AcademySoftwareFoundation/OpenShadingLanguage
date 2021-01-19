// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

//#define OSL_DEV

#include <boost/container/flat_set.hpp>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <llvm/ADT/SmallVector.h>

#include <OSL/batched_rendererservices.h>

#include "batched_analysis.h"
#include "oslexec_pvt.h"

using namespace OSL;
using namespace OSL::pvt;


OSL_NAMESPACE_ENTER

namespace Strings {

// TODO: What qualifies these to move to strdecls.h?
//       Being used in more than one .cpp?
// Operation strings
static ustring op_and("and");
static ustring op_backfacing("backfacing");
static ustring op_break("break");
static ustring op_calculatenormal("calculatenormal");
static ustring op_compl("compl");
static ustring op_concat("concat");
static ustring op_continue("continue");
static ustring op_endswith("endswith");
static ustring op_eq("eq");
static ustring op_functioncall("functioncall");
static ustring op_functioncall_nr("functioncall_nr");
static ustring op_ge("ge");
static ustring op_getattribute("getattribute");
static ustring op_getchar("getchar");
static ustring op_getmatrix("getmatrix");
static ustring op_getmessage("getmessage");
static ustring op_gt("gt");
static ustring op_hash("hash");
static ustring op_if("if");
static ustring op_le("le");
static ustring op_lt("lt");
static ustring op_neq("neq");
static ustring op_or("or");
static ustring op_pow("pow");
static ustring op_return("return");
static ustring op_startswith("startswith");
static ustring op_stoi("stoi");
static ustring op_stof("stof");
static ustring op_strlen("strlen");
static ustring op_substr("substr");
static ustring op_surfacearea("surfacearea");
static ustring op_trace("trace");
static ustring op_transform("transform");
static ustring op_transformv("transformv");
static ustring op_transformn("transformn");
static ustring op_texture("texture");
static ustring op_texture3d("texture3d");

// Shader global strings
static ustring object2common("object2common");
static ustring shader2common("shader2common");
static ustring flipHandedness("flipHandedness");
}  // namespace Strings

namespace pvt {

// Implemented in batched_backendllvm.cpp
extern bool
is_shader_global_uniform_by_name(ustring name);


// When adding new ops, or when changing implementation or behavior of
// existing ops, the following functions should be evaluated to see if
// they need to be updated to handle to handle the new op or change:
//     bool are_op_results_always_implicitly_varying(ustring opname);
//     bool are_op_results_always_implicitly_uniform(ustring opname);
//     bool does_op_implementation_require_masking(ustring opname);
//     SymbolPtrVec * protected_shader_globals_op_implicitly_depends_on(ShaderInstance &inst, Opcode & opcode);
// NOTE: psg stands for ProtectedShaderGlobals which are explained below...

// If there is a new or change in control flow op's are made,
// a more detailed evaluation will be needed as their behavior
// is hard coded below vs. lookup based implementation of the
// functions above.

namespace  // Unnamed
{
// Even when all inputs to an operation are uniform,
// the results written to output symbols maybe varying.
bool
are_op_results_always_implicitly_varying(ustring opname)
{
    return (opname == Strings::op_getmessage) | (opname == Strings::op_trace)
           | (opname == Strings::op_texture)
           | (opname == Strings::op_texture3d);
    // Renderer might identify result of getattribute as always uniform
    // depending on the attribute itself, so it cannot
    // be "always" implicitly varying based solely on the opname.
    // We consider getattribute during discovery.
}

// Even when all inputs to an operation are varying,
// the results written to output symbols maybe uniform.
bool
are_op_results_always_implicitly_uniform(ustring opname)
{
    // arraylength(val) will always have a uniform
    // result as only fixed size arrays are supported.
    return (opname == Strings::arraylength);
}


// Analysis may determine that the results of an operation need
// not be masked.  However the implementation of the op may require masking
// for performance reasons or in the case only a mask implementation
// was provided to minimize number of library functions.
// NOTE: For operations that llvmgen is hardcoded to use masking
// (essentially ignoring Opcode::requires_masking()), their opnames do NOT
// need to be identified here.  However any op's using the generic llvmgen
// whose implementation does require masking WILL need to be here to enable
// llvmgen to dispatch to the correct function name and pass a mask
bool
does_op_implementation_require_masking(ustring opname)
{
    // TODO: should OpDescriptor handle identifying operations that always
    // require masking vs. this lazy_lookup?  Perhaps a BatchedOpDescriptor?
    static boost::container::flat_set<ustring> lazy_lookup(
        { // safe_pows's implementation uses an OSL_UNLIKELY
          // which will perform a horizontal operation to check if
          // a condition is false for all data lane in order to skip
          // over an expensive code block.  Even though storing the
          // result value doesn't require masking,
          // we still need to pass a mask so that inactive lanes
          // can be excluded from the horizontal check.
          // If this was not done, the results will still be correct
          // however the expensive code block could execute needlessly
          Strings::op_pow,
          // TODO: more operations probably need to be added to this list,
          // in particular any whose implementations use OSL_LIKELY or
          // OSL_UNLIKELY

          // Implementation decision for all string manipulation
          // functions to require masks
          Strings::op_concat, Strings::op_strlen, Strings::op_hash,
          Strings::op_getchar, Strings::op_startswith, Strings::op_endswith,
          Strings::op_stoi, Strings::op_stof, Strings::op_substr });
    return lazy_lookup.find(opname) != lazy_lookup.end();
}

// Create some placeholder symbols for shader globals that have don't
// normally have symbols.
// Purpose is to allow our data structures to use Symbol * as a key
struct Symbols4ProtectedShaderGlobals {
    Symbol shader2common;
    Symbol object2common;
    Symbol flipHandedness;
    Symbol raytype;
    Symbol backfacing;
    Symbol surfacearea;
    Symbol time;

    SymbolPtrVec varying_symbols;

    void check_if_varing(Symbol& s)
    {
        if (!is_shader_global_uniform_by_name(s.name())) {
            s.make_varying();
            varying_symbols.push_back(&s);
        }
    }

    Symbols4ProtectedShaderGlobals()
        : shader2common(Strings::shader2common, TypeSpec(), SymTypeGlobal)
        , object2common(Strings::object2common, TypeSpec(), SymTypeGlobal)
        , flipHandedness(Strings::flipHandedness, TypeSpec(), SymTypeGlobal)
        , raytype(Strings::raytype, TypeSpec(), SymTypeGlobal)
        , backfacing(Strings::op_backfacing, TypeSpec(), SymTypeGlobal)
        , surfacearea(Strings::op_surfacearea, TypeSpec(), SymTypeGlobal)
        , time(Strings::time, TypeSpec(), SymTypeGlobal)
    {
        check_if_varing(shader2common);
        check_if_varing(object2common);
        check_if_varing(flipHandedness);
        check_if_varing(raytype);
        check_if_varing(backfacing);
        check_if_varing(surfacearea);
        check_if_varing(time);
    }
};

Symbols4ProtectedShaderGlobals&
psg()
{
    static Symbols4ProtectedShaderGlobals lazy_psg;
    return lazy_psg;
}


// Helper class to manage lookups of which Protected Shader Globals
// a specific op implicitly depends on
class ImplicitDependenciesOnPsgs {
private:
    typedef bool (*OpcodeTest)(ShaderInstance&, Opcode&);

    struct Dependency {
        // Optional test for filtering based on opcode arguments
        OpcodeTest test;
        // Set of protected shader global symbols that opcode
        // implicitly depends upon
        SymbolPtrVec symbols;
    };

    // When Dependency.test == no_test, no additional test is required
    static constexpr OpcodeTest no_test = nullptr;

    // Map opcode names to a Dependency info
    std::unordered_map<ustring, Dependency, ustringHash> m_lookup;


    /// Return the ptr to the symbol that is the argnum-th argument to the
    /// given op in the instance.
    static Symbol* opargsym(ShaderInstance& inst, const Opcode& op, int argnum)
    {
        return (argnum < op.nargs()) ? inst.argsymbol(op.firstarg() + argnum)
                                     : nullptr;
    };

    // Specific tests to see if opcode's whose arguments affect whether
    // they have implicit dependencies on protected shader globals or not
    static bool is_transform_using_space(ShaderInstance& inst, Opcode& opcode)
    {
        int arg_count    = opcode.nargs();
        auto To          = opargsym(inst, opcode, (arg_count == 3) ? 1 : 2);
        bool using_space = (false == To->typespec().is_matrix());
        if (using_space) {
            // if From and To strings are the same at JIT time
            // we don't need space
            auto From = (arg_count == 3) ? NULL : opargsym(inst, opcode, 1);
            if ((From == NULL || From->is_constant()) && To->is_constant()) {
                // We can know all the space names at this time
                ustring from = From ? *((ustring*)From->data())
                                    : Strings::common;
                ustring to  = *((ustring*)To->data());
                ustring syn = inst.shadingsys().commonspace_synonym();
                if (from == syn)
                    from = Strings::common;
                if (to == syn)
                    to = Strings::common;
                if (from == to) {
                    using_space = false;
                }
            }
        }
        return using_space;
    };

    static bool is_triple_using_space(ShaderInstance& inst, Opcode& opcode)
    {
        int arg_count    = opcode.nargs();
        bool using_space = (arg_count == 5);
        if (using_space) {
            auto Space = opargsym(inst, opcode, 1);
            if (Space->is_constant()) {
                auto from = Space->get_string();
                if (from == Strings::common
                    || from == inst.shadingsys().commonspace_synonym()) {
                    using_space = false;  // no transformation necessary
                }
            }
        }
        return using_space;
    };

    static bool is_matrix_using_space(ShaderInstance& inst, Opcode& opcode)
    {
        int arg_count = opcode.nargs();
        // Only certain variations of matrix use shader globals
        bool using_space = (arg_count == 3 || arg_count == 18);
        bool using_two_spaces
            = (arg_count == 3
               && opargsym(inst, opcode, 2)->typespec().is_string());
        if (using_two_spaces)
            return true;
        if (using_space) {
            auto Space = opargsym(inst, opcode, 1);
            if (Space->is_constant()) {
                auto from = Space->get_string();
                if (from == Strings::common
                    || from == inst.shadingsys().commonspace_synonym()) {
                    using_space = false;  // no transformation necessary
                }
            }
        }
        return using_space;
    };

public:
    ImplicitDependenciesOnPsgs()
    {
        // Initialize lookup map
        using namespace Strings;
        m_lookup[op_calculatenormal] = Dependency { no_test,
                                                    { &psg().flipHandedness } };
        m_lookup[raytype]        = Dependency { no_test, { &psg().raytype } };
        m_lookup[op_surfacearea] = Dependency { no_test,
                                                { &psg().surfacearea } };
        m_lookup[op_backfacing] = Dependency { no_test, { &psg().backfacing } };

        SymbolPtrVec syms_4_space(
            { &psg().shader2common, &psg().object2common, &psg().time });

        m_lookup[op_transform]  = Dependency { &is_transform_using_space,
                                              syms_4_space };
        m_lookup[op_transformn] = Dependency { &is_transform_using_space,
                                               syms_4_space };
        m_lookup[op_transformv] = Dependency { &is_transform_using_space,
                                               syms_4_space };
        m_lookup[vector] = Dependency { &is_triple_using_space, syms_4_space };
        m_lookup[point]  = Dependency { &is_triple_using_space, syms_4_space };
        m_lookup[normal] = Dependency { &is_triple_using_space, syms_4_space };
        m_lookup[op_getmatrix] = Dependency { no_test, syms_4_space };
        m_lookup[matrix] = Dependency { &is_matrix_using_space, syms_4_space };
    }

    SymbolPtrVec* lookup(ShaderInstance& inst, Opcode& opcode)
    {
        auto iter = m_lookup.find(opcode.opname());
        if (iter != m_lookup.end()) {
            auto& dependency = iter->second;
            if ((dependency.test == no_test)
                || (*dependency.test)(inst, opcode)) {
                return &dependency.symbols;
            }
        }
        return nullptr;
    }
};

static SymbolPtrVec*
protected_shader_globals_op_implicitly_depends_on(ShaderInstance& inst,
                                                  Opcode& opcode)
{
    static ImplicitDependenciesOnPsgs lazyDependencies;
    return lazyDependencies.lookup(inst, opcode);
}


bool
is_op_result_always_logically_boolean(ustring opname)
{
    // Test if explicit comparison is faster or not
    static boost::container::flat_set<ustring> lazy_lookup(
        { Strings::op_getattribute, Strings::op_compl, Strings::op_eq,
          Strings::op_ge, Strings::op_gt, Strings::op_le, Strings::op_lt,
          Strings::op_neq, Strings::op_and, Strings::op_or });
    return lazy_lookup.find(opname) != lazy_lookup.end();
}



// The Position returned by top_pos changes and symbols are pushed and popped.
// However any given position is invariant as scopes change, and one
// can iterate over any previous representation of the dependency stack by
// calling begin_at(pos).
// Idea is the top_pos can be cached per instruction and later be used
// to iterate over the stack of dependent symbols at for that instruction.
// This should be much cheaper than keeping a unique list per instruction.
// Nothing invalidates an iterator.
class DependencyTreeTracker {
public:
    // Simple wrapper to improve readability
    class Position {
        int m_index;

    public:
        explicit OSL_FORCEINLINE Position(int node_index) : m_index(node_index)
        {
        }

        Position()                = default;
        Position(const Position&) = default;

        OSL_FORCEINLINE int operator()() const { return m_index; }

        bool operator==(const Position& other) const
        {
            return m_index == other.m_index;
        }

        bool operator!=(const Position& other) const
        {
            return m_index != other.m_index;
        }
    };

    static OSL_FORCEINLINE Position end_pos() { return Position(-1); }

private:
    struct Node {
        Node(Position parent_, int depth_, Symbol* sym_)
            : parent(parent_), depth(depth_), sym(sym_)
        {
        }

        Position parent;
        int depth;
        Symbol* sym;
    };

    std::vector<Node> m_nodes;
    Position m_top_of_stack;
    int m_current_depth;

public:
    DependencyTreeTracker() : m_top_of_stack(end_pos()), m_current_depth(0) {}

    DependencyTreeTracker(const DependencyTreeTracker&) = delete;


    class Iterator {
        Position m_pos;
        const DependencyTreeTracker* m_dtt;

        OSL_FORCEINLINE const Node& node() const
        {
            return m_dtt->m_nodes[m_pos()];
        }

    public:
        typedef Symbol* value_type;
        typedef int difference_type;
        // read only data, no intention of giving a reference out
        typedef Symbol* reference;
        typedef Symbol* pointer;
        typedef std::forward_iterator_tag iterator_category;

        OSL_FORCEINLINE
        Iterator() : m_dtt(nullptr) {}

        OSL_FORCEINLINE explicit Iterator(const DependencyTreeTracker& dtt,
                                          Position pos)
            : m_pos(pos), m_dtt(&dtt)
        {
        }

        OSL_FORCEINLINE Position pos() const { return m_pos; };

        OSL_FORCEINLINE int depth() const
        {
            // Make sure we didn't try to access the end
            if (m_pos() == end_pos()())
                return 0;
            return node().depth;
        };

        OSL_FORCEINLINE Iterator& operator++()
        {
            // prefix operator
            m_pos = node().parent;
            return *this;
        }

        OSL_FORCEINLINE Iterator operator++(int)
        {
            // postfix operator
            Iterator retVal(*this);
            m_pos = node().parent;
            return retVal;
        }

        OSL_FORCEINLINE Symbol* operator*() const
        {
            // Make sure we didn't try to access the end
            OSL_ASSERT(m_pos() != end_pos()());
            return node().sym;
        }

        OSL_FORCEINLINE bool operator==(const Iterator& other)
        {
            return m_pos() == other.m_pos();
        }

        OSL_FORCEINLINE bool operator!=(const Iterator& other)
        {
            return m_pos() != other.m_pos();
        }
    };

    // Validate that the Iterator meets the requirements of a std::forward_iterator_tag
    static_assert(
        std::is_default_constructible<Iterator>::value,
        "DependencyTreeTracker::Iterator must be default constructible");
    static_assert(std::is_copy_constructible<Iterator>::value,
                  "DependencyTreeTracker::Iterator must be copy constructible");
    static_assert(std::is_copy_assignable<Iterator>::value,
                  "DependencyTreeTracker::Iterator must be copy assignable");
    static_assert(std::is_move_assignable<Iterator>::value,
                  "DependencyTreeTracker::Iterator must be move assignable");
    static_assert(std::is_destructible<Iterator>::value,
                  "DependencyTreeTracker::Iterator must be destructible");
    static_assert(std::is_same<decltype(std::declval<Iterator>()
                                        == std::declval<Iterator>()),
                               bool>::value,
                  "DependencyTreeTracker::Iterator must be equality comparable");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::value_type,
                     Symbol*>::value,
        "DependencyTreeTracker::Iterator must define type value_type");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::difference_type,
                     int>::value,
        "DependencyTreeTracker::Iterator must define type difference_type");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::reference,
                     Symbol*>::value,
        "DependencyTreeTracker::Iterator must define type reference");
    static_assert(std::is_same<typename std::iterator_traits<Iterator>::pointer,
                               Symbol*>::value,
                  "DependencyTreeTracker::Iterator must define type pointer");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::iterator_category,
                     std::forward_iterator_tag>::value,
        "DependencyTreeTracker::Iterator must define type iterator_category");
    static_assert(
        std::is_same<decltype(*std::declval<Iterator>()), Symbol*>::value,
        "DependencyTreeTracker::Iterator must implement reference operator *");
    static_assert(
        std::is_same<decltype(++std::declval<Iterator>()), Iterator&>::value,
        "DependencyTreeTracker::Iterator must implement Iterator & operator ++");
    static_assert(
        std::is_same<decltype((void)std::declval<Iterator>()++), void>::value,
        "DependencyTreeTracker::Iterator must implement Iterator & operator ++ (int)");
    static_assert(
        std::is_same<decltype(*std::declval<Iterator>()++), Symbol*>::value,
        "DependencyTreeTracker::Iterator must support *it++");
    static_assert(
        std::is_same<decltype(std::declval<Iterator>()
                              != std::declval<Iterator>()),
                     bool>::value,
        "DependencyTreeTracker::Iterator must implement bool operator != (const Iterator &)");
    static_assert(
        std::is_same<decltype(!(std::declval<Iterator>()
                                == std::declval<Iterator>())),
                     bool>::value,
        "DependencyTreeTracker::Iterator must implement bool operator == (const Iterator &)");

    OSL_FORCEINLINE Iterator begin() const
    {
        return Iterator(*this, top_pos());
    }

    OSL_FORCEINLINE Iterator begin_at(Position pos) const
    {
        return Iterator(*this, pos);
    }

    OSL_FORCEINLINE Iterator end() const { return Iterator(*this, end_pos()); }

    OSL_FORCEINLINE void push(Symbol* sym)
    {
        ++m_current_depth;

        Position parent(m_top_of_stack);
        Node node(parent, m_current_depth, sym);
        m_top_of_stack = Position(static_cast<int>(m_nodes.size()));
        m_nodes.push_back(node);
    }

    OSL_FORCEINLINE Position top_pos() const { return m_top_of_stack; }

    OSL_FORCEINLINE const Symbol* top() const
    {
        return m_nodes[m_top_of_stack()].sym;
    }

    void pop()
    {
        OSL_ASSERT(m_current_depth > 0);
        OSL_ASSERT(m_top_of_stack() != end_pos()());
        m_top_of_stack = m_nodes[m_top_of_stack()].parent;
        --m_current_depth;
    }



    bool is_descendent_or_self(Position pos, Position potential_ancestor)
    {
        auto endAt = end();
        auto iter  = begin_at(pos);
        // allow testing of pos == potential_ancestor when potential_ancestor == end_pos()
        do {
            if (iter.pos() == potential_ancestor) {
                return true;
            }
        } while (iter++ != endAt);
        return false;
    }

    Position common_ancestor_between(Position read_pos, Position write_pos)
    {
        // To find common anscenstor
        // Walk up both paths building 2 stacks of positions
        // now iterate from tops of stack while the positions are the same.
        // When the differ the previous position was the common ancestor
        auto read_iter  = begin_at(read_pos);
        auto write_iter = begin_at(write_pos);

        const int read_count = read_iter.depth() + 1;
        OSL_STACK_ARRAY(Position, read_path, read_count);
        int read_depth = 0;
        {
            // Loop structure is to allow
            // writing the end position into the read path
            for (;;) {
                OSL_ASSERT(read_depth < read_count);
                read_path[read_depth++] = read_iter.pos();
                if (read_iter == end())
                    break;
                ++read_iter;
            }
        }

        const int writeCount = write_iter.depth() + 1;
        OSL_STACK_ARRAY(Position, write_path, writeCount);
        int write_depth = 0;
        {
            // Loop structure is to allow
            // writing the end position into the write path
            for (;;) {
                OSL_ASSERT(write_depth < writeCount);
                write_path[write_depth++] = write_iter.pos();
                if (write_iter == end())
                    break;
                ++write_iter;
            }
        }
        const int levelCount  = std::min(read_depth, write_depth);
        const int read_level  = read_depth - 1;
        const int write_level = write_depth - 1;
        OSL_ASSERT(write_path[write_level] == end_pos());
        OSL_ASSERT(read_path[read_level] == end_pos());



        int level = 0;
        OSL_ASSERT(read_path[read_level - level]
                   == write_path[write_level - level]);
        do {
            ++level;
            if (read_path[read_level - level]()
                != write_path[write_level - level]())
                break;
        } while (level < levelCount);
        --level;
        OSL_ASSERT(read_path[read_level - level]
                   == write_path[write_level - level]);
        OSL_ASSERT(read_level - level >= 0);
        return read_path[read_level - level];
    }
};

// Historically tracks stack of functions
// The Position returned by top_pos changes and function scopes are pushed and popped.
// However any given position is invariant as scopes change
// Idea is the top_pos can be cached per instruction and later be iterated
class FunctionTreeTracker {
public:
    // Simple wrapper to improve readability
    class Position {
        int m_index;

    public:
        explicit OSL_FORCEINLINE Position(int node_index) : m_index(node_index)
        {
        }

        OSL_FORCEINLINE Position()
        { /* uninitialzied */
        }

        Position(const Position&) = default;

        OSL_FORCEINLINE int operator()() const { return m_index; }

        OSL_FORCEINLINE bool operator==(const Position& other) const
        {
            return m_index == other.m_index;
        }

        OSL_FORCEINLINE bool operator!=(const Position& other) const
        {
            return m_index != other.m_index;
        }
    };

    static OSL_FORCEINLINE Position end_pos() { return Position(-1); }

    enum Type { TypeReturn, TypeExit };

    struct EarlyOut {
        explicit OSL_FORCEINLINE
        EarlyOut(Type type_, DependencyTreeTracker::Position dtt_pos_)
            : type(type_), dtt_pos(dtt_pos_)
        {
        }

        Type type;
        DependencyTreeTracker::Position dtt_pos;
    };

private:
    struct Node {
        explicit OSL_FORCEINLINE Node(Position parent_,
                                      const EarlyOut& early_out_)
            : parent(parent_), early_out(early_out_)
        {
        }

        Position parent;
        EarlyOut early_out;
    };

    std::vector<Node> m_nodes;
    Position m_top_of_stack;
    std::vector<Position> m_function_stack;
    std::vector<Position> m_before_if_block_stack;
    std::vector<Position> m_after_if_block_stack;

public:
    OSL_FORCEINLINE FunctionTreeTracker() : m_top_of_stack(end_pos())
    {
        push_function_call();
    }

    FunctionTreeTracker(const FunctionTreeTracker&) = delete;


    class Iterator {
        Position m_pos;
        const FunctionTreeTracker* m_ftt;

        OSL_FORCEINLINE const Node& node() const
        {
            return m_ftt->m_nodes[m_pos()];
        }

    public:
        typedef const EarlyOut& value_type;
        typedef int difference_type;
        // read only data, no intention of giving a reference out
        typedef const EarlyOut& reference;
        typedef const EarlyOut* pointer;
        typedef std::forward_iterator_tag iterator_category;

        OSL_FORCEINLINE
        Iterator() : m_ftt(nullptr) {}

        OSL_FORCEINLINE explicit Iterator(const FunctionTreeTracker& ftt,
                                          Position pos)
            : m_pos(pos), m_ftt(&ftt)
        {
        }

        OSL_FORCEINLINE Position pos() const { return m_pos; };

        OSL_FORCEINLINE Iterator& operator++()
        {
            // prefix operator
            m_pos = node().parent;
            return *this;
        }

        OSL_FORCEINLINE Iterator operator++(int)
        {
            // postfix operator
            Iterator retVal(*this);
            m_pos = node().parent;
            return retVal;
        }

        OSL_FORCEINLINE const EarlyOut& operator*() const
        {
            // Make sure we didn't try to access the end
            OSL_ASSERT(m_pos() != end_pos()());
            return node().early_out;
        }

        OSL_FORCEINLINE bool operator==(const Iterator& other)
        {
            return m_pos() == other.m_pos();
        }

        OSL_FORCEINLINE bool operator!=(const Iterator& other)
        {
            return m_pos() != other.m_pos();
        }
    };

    // Validate that the Iterator meets the requirements of a std::forward_iterator_tag
    static_assert(std::is_default_constructible<Iterator>::value,
                  "FunctionTreeTracker::Iterator must be default constructible");
    static_assert(std::is_copy_constructible<Iterator>::value,
                  "FunctionTreeTracker::Iterator must be copy constructible");
    static_assert(std::is_copy_assignable<Iterator>::value,
                  "FunctionTreeTracker::Iterator must be copy assignable");
    static_assert(std::is_move_assignable<Iterator>::value,
                  "FunctionTreeTracker::Iterator must be move assignable");
    static_assert(std::is_destructible<Iterator>::value,
                  "FunctionTreeTracker::Iterator must be destructible");
    static_assert(std::is_same<decltype(std::declval<Iterator>()
                                        == std::declval<Iterator>()),
                               bool>::value,
                  "FunctionTreeTracker::Iterator must be equality comparable");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::value_type,
                     const EarlyOut&>::value,
        "FunctionTreeTracker::Iterator must define type value_type");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::difference_type,
                     int>::value,
        "FunctionTreeTracker::Iterator must define type difference_type");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::reference,
                     const EarlyOut&>::value,
        "FunctionTreeTracker::Iterator must define type reference");
    static_assert(std::is_same<typename std::iterator_traits<Iterator>::pointer,
                               const EarlyOut*>::value,
                  "FunctionTreeTracker::Iterator must define type pointer");
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::iterator_category,
                     std::forward_iterator_tag>::value,
        "FunctionTreeTracker::Iterator must define type iterator_category");
    static_assert(
        std::is_same<decltype(*std::declval<Iterator>()), const EarlyOut&>::value,
        "FunctionTreeTracker::Iterator must implement reference operator *");
    static_assert(
        std::is_same<decltype(++std::declval<Iterator>()), Iterator&>::value,
        "FunctionTreeTracker::Iterator must implement Iterator & operator ++");
    static_assert(
        std::is_same<decltype((void)std::declval<Iterator>()++), void>::value,
        "FunctionTreeTracker::Iterator must implement Iterator & operator ++ (int)");
    static_assert(std::is_same<decltype(*std::declval<Iterator>()++),
                               const EarlyOut&>::value,
                  "FunctionTreeTracker::Iterator must support *it++");
    static_assert(
        std::is_same<decltype(std::declval<Iterator>()
                              != std::declval<Iterator>()),
                     bool>::value,
        "FunctionTreeTracker::Iterator must implement bool operator != (const Iterator &)");
    static_assert(
        std::is_same<decltype(!(std::declval<Iterator>()
                                == std::declval<Iterator>())),
                     bool>::value,
        "FunctionTreeTracker::Iterator must implement bool operator == (const Iterator &)");

    OSL_FORCEINLINE Iterator begin() const
    {
        return Iterator(*this, top_pos());
    }

    OSL_FORCEINLINE Iterator begin_at(Position pos) const
    {
        return Iterator(*this, pos);
    }

    OSL_FORCEINLINE Iterator end() const { return Iterator(*this, end_pos()); }

    OSL_FORCEINLINE void push_function_call()
    {
        OSL_DEV_ONLY(std::cout << "DependencyTreeTracker push_function_call"
                               << std::endl);
        m_function_stack.push_back(m_top_of_stack);
    }

    OSL_FORCEINLINE void push_if_block()
    {
        // hold onto the position that existed at the beginning of the if block
        // because that is the set of early outs that the else block should
        // use, any early outs coming from the if block should be ignored for the
        // else block
        m_before_if_block_stack.push_back(m_top_of_stack);
    }

    OSL_FORCEINLINE void pop_if_block()
    {
        OSL_ASSERT(!m_before_if_block_stack.empty());
        // we defer actually popping the stack until the else is
        // popped, it is a package deal "if+else" always
    }

    OSL_FORCEINLINE void push_else_block()
    {
        // Remember the current top pos, to restore and the end of the else block
        m_after_if_block_stack.push_back(m_top_of_stack);

        // Now use only the early outs that existed
        // at the beginning of the if block, essentially ignoring any early outs
        // processed inside the the if block
        m_top_of_stack = m_before_if_block_stack.back();
    }
    OSL_FORCEINLINE void pop_else_block()
    {
        Position pos_after_else = m_top_of_stack;

        Position pos_after_if = m_after_if_block_stack.back();
        m_after_if_block_stack.pop_back();

        // Restore the early outs to the state after the if block
        m_top_of_stack = pos_after_if;

        // Add on any early outs processed inside the else block
        Position pos_before_else = m_before_if_block_stack.back();
        m_before_if_block_stack.pop_back();

        // NOTE: as we can only walk the tree upstream, the order
        // of early outs will be reverse the original.
        // The algorithm utilizes only the set of early outs and
        // order should not matter.  If the algorithm changes
        // this may need to be revisited.
        auto end_iter = begin_at(pos_before_else);
        for (auto iter = begin_at(pos_after_else); iter != end_iter; ++iter) {
            const EarlyOut& eo = *iter;
            if (eo.type == TypeReturn) {
                process_return(eo.dtt_pos);
            } else {
                process_exit(eo.dtt_pos);
            }
        }
    }

    OSL_FORCEINLINE void process_return(DependencyTreeTracker::Position dttPos)
    {
        OSL_DEV_ONLY(std::cout << "DependencyTreeTracker process_return"
                               << std::endl);
        Position parent(m_top_of_stack);
        Node node(parent, EarlyOut(TypeReturn, dttPos));
        m_top_of_stack = Position(static_cast<int>(m_nodes.size()));
        m_nodes.push_back(node);
    }

    OSL_FORCEINLINE void process_exit(DependencyTreeTracker::Position dttPos)
    {
        OSL_DEV_ONLY(std::cout << "DependencyTreeTracker process_exit"
                               << std::endl);
        Position parent(m_top_of_stack);
        Node node(parent, EarlyOut(TypeExit, dttPos));
        m_top_of_stack = Position(static_cast<int>(m_nodes.size()));
        m_nodes.push_back(node);
    }


    OSL_FORCEINLINE Position top_pos() const { return m_top_of_stack; }

    OSL_FORCEINLINE bool has_early_out() const
    {
        return m_top_of_stack != end_pos();
    }

    OSL_FORCEINLINE void pop_function_call()
    {
        OSL_DEV_ONLY(std::cout << "DependencyTreeTracker pop_function_call"
                               << std::endl);

        OSL_ASSERT(false == m_function_stack.empty());

        // Because iterators are never invalidated, we can walk through a chain of early outs
        // while actively modifying the stack
        auto earlyOutIter         = begin();
        Position posAtStartOfFunc = m_function_stack.back();
        m_function_stack.pop_back();

        // popped all early outs back to the state at the when the function was pushed
        m_top_of_stack = posAtStartOfFunc;

        auto end_of_early_outs = begin_at(posAtStartOfFunc);

        for (; earlyOutIter != end_of_early_outs; ++earlyOutIter) {
            const EarlyOut& early_out = *earlyOutIter;
            if (early_out.type == TypeExit) {
                // exits are sticky, we need to append the exit onto
                // the calling function
                // And since we already moved the top of the stack to the state the calling
                // function was at, we can just
                process_exit(early_out.dtt_pos);
            }
        }
    }
};

class WriteEvent {
    DependencyTreeTracker::Position m_pos_in_tree;
    int m_op_num;
    int m_loop_op_index;
    static constexpr int InitialAssignmentOp() { return -1; }
    static constexpr int NoLoopIndex() { return -1; }

public:
    WriteEvent(DependencyTreeTracker::Position pos_in_tree_, int op_num_,
               int loop_op_index)
        : m_pos_in_tree(pos_in_tree_)
        , m_op_num(op_num_)
        , m_loop_op_index(loop_op_index)
    {
    }

    explicit WriteEvent(DependencyTreeTracker::Position pos_in_tree_)
        : m_pos_in_tree(pos_in_tree_)
        , m_op_num(InitialAssignmentOp())
        , m_loop_op_index(NoLoopIndex())
    {
    }

    OSL_FORCEINLINE DependencyTreeTracker::Position pos_in_tree() const
    {
        return m_pos_in_tree;
    }

    OSL_FORCEINLINE bool is_initial_assignment() const
    {
        return m_op_num == InitialAssignmentOp();
    }

    OSL_FORCEINLINE int op_num() const
    {
        OSL_ASSERT(!is_initial_assignment());
        return m_op_num;
    }


    OSL_FORCEINLINE int loop_op_index() const { return m_loop_op_index; }
};

typedef std::vector<WriteEvent> WriteChronology;
//typedef llvm::SmallVector<WriteEvent, 8> WriteChronology;


class ReadEvent {
    DependencyTreeTracker::Position m_pos_in_tree;
    int m_op_num;
    int m_loop_op_index;
    static constexpr int InitialReadOp() { return -1; }
    static constexpr int NoLoopIndex() { return -1; }

public:
    ReadEvent(DependencyTreeTracker::Position pos_in_tree_
#ifdef OSL_DEV
              ,
              int op_num_, int loop_op_index
#endif
              )
        : m_pos_in_tree(pos_in_tree_)
#ifdef OSL_DEV
        , m_op_num(op_num_)
        , m_loop_op_index(loop_op_index)
#endif
    {
    }

    OSL_FORCEINLINE DependencyTreeTracker::Position pos_in_tree() const
    {
        return m_pos_in_tree;
    }

#ifdef OSL_DEV
    OSL_FORCEINLINE int op_num() const { return m_op_num; }

    OSL_FORCEINLINE int loop_op_index() const { return m_loop_op_index; }
#endif
};

typedef std::vector<ReadEvent> ReadChronology;
// Testing showed 6 read events a typically max
//typedef llvm::SmallVector<ReadEvent, 6> ReadChronology;

class LoopStack {
    std::vector<int> m_conditions_op_index_stack;
    std::vector<const Symbol*> m_conditions_symbol_stack;
    std::unordered_map<const Symbol*, ReadChronology>
        m_potentiallyCyclicReadOpsBySymbol;

    struct LoopInfo {
        int controlFlowOpIndex;
        Symbol* controlFlowSymbol;

        LoopInfo(int loop_op_index, Symbol* condition_variable)
            : controlFlowOpIndex(loop_op_index)
            , controlFlowSymbol(condition_variable)
        {
        }
    };

    std::vector<LoopInfo> m_loop_info_by_depth;

public:
    bool is_inside_loop() const { return !m_loop_info_by_depth.empty(); }

    int depth() const { return m_loop_info_by_depth.size(); }

    int current_loop_op_index() const
    {
        if (m_loop_info_by_depth.empty())
            return -1;
        return m_loop_info_by_depth.back().controlFlowOpIndex;
    };

    Symbol* current_condition() const
    {
        OSL_ASSERT(is_inside_loop());
        return m_loop_info_by_depth.back().controlFlowSymbol;
    };

    void push_loop(int loop_op_index, Symbol* condition_variable)
    {
        m_loop_info_by_depth.emplace_back(
            LoopInfo(loop_op_index, condition_variable));
    };

    void pop_loop(int loop_op_index)
    {
        OSL_ASSERT(is_inside_loop());
        m_loop_info_by_depth.pop_back();
        if (!is_inside_loop()) {
            // We are no longer in a loop, so no chance for the
            // reads to execute again, so we can stop tracking them
            m_potentiallyCyclicReadOpsBySymbol.clear();
        }
    };

    void
    track_potential_cyclic_read(const Symbol* read_sym,
                                DependencyTreeTracker::Position pos_in_tree_,
                                int op_num_)
    {
        OSL_ASSERT(is_inside_loop());
        m_potentiallyCyclicReadOpsBySymbol[read_sym].emplace_back(
            ReadEvent(pos_in_tree_
#ifdef OSL_DEV
                      ,
                      op_num_, current_loop_op_index()
#endif
                          ));
    }

    const ReadChronology& potentiallyCyclicReadsFor(const Symbol* sym)
    {
        auto lookup = m_potentiallyCyclicReadOpsBySymbol.find(sym);
        if (lookup != m_potentiallyCyclicReadOpsBySymbol.end()) {
            return lookup->second;
        } else {
            static ReadChronology emptyReadChronology;
            return emptyReadChronology;
        }
    }
};

struct ReadBranch {
    DependencyTreeTracker::Position pos;
    int depth;  // how deep in the m_conditional_symbol_stack did the read occur

    ReadBranch()
        : pos(DependencyTreeTracker::end_pos())
        , depth(std::numeric_limits<int>::max())
    {
    }
};

struct Analyzer {
    BatchedAnalysis& m_ba;
    ShaderInstance* m_inst;
    OpcodeVec& m_opcodes;
    const int m_op_count;
    std::unordered_map<const Symbol*, WriteChronology>
        m_write_chronology_by_symbol;
    DependencyTreeTracker m_conditional_symbol_stack;
    FunctionTreeTracker m_execution_scope_stack;

    // Track the shallowest ancestor that reads the results of a particular operation.
    // We will need to establish dependencies to any conditionals
    // that exist between the branch to the shallowest reader and the opcode's position
    // in the m_conditional_symbol_stack
    std::vector<ReadBranch> m_shallowest_read_branch_by_op_index;
    LoopStack m_loop_stack;

    std::vector<DependencyTreeTracker::Position>
        m_pos_in_dependent_sym_stack_by_op_index;
    std::vector<FunctionTreeTracker::Position>
        m_pos_in_exec_scope_stack_by_op_index;

    std::vector<int> m_deferred_op_indices_to_be_masked;

    // Model all symbols downstream that given symbol affects.
    // Once built, we use this to push "varying" downstream from
    // shader globals through all operations that take their symbols as inputs
    // and feed them forward to those operations outputs (then repeat
    // recursively for those outputs)
    std::unordered_multimap<Symbol* /* parent */, Symbol* /* dependent */,
                            std::hash<Symbol*>, std::equal_to<Symbol*>>
        m_symbols_dependent_upon;

    std::unordered_set<Symbol*> m_symbols_written_to_by_implicitly_varying_ops;


    // Remember which symbols we've force to be boolean so we can
    // reverse that decision if we see them used by any boolean operations
    std::unordered_set<Symbol*> m_symbols_logically_bool;
    std::unordered_set<Symbol*> m_symbols_disqualified_from_bool;

    ShaderInstance* inst() const { return m_inst; }
    Opcode& op(int opnum) { return m_opcodes[opnum]; }
    /// Return the ptr to the symbol that is the argnum-th argument to the
    /// given op in the current instance.
    Symbol* opargsym(const Opcode& op, int argnum)
    {
        return (argnum < op.nargs()) ? inst()->argsymbol(op.firstarg() + argnum)
                                     : nullptr;
    };

    Analyzer(BatchedAnalysis& batched_analysis, ShaderInstance* inst)
        : m_ba(batched_analysis)
        , m_inst(inst)
        , m_opcodes(inst->ops())
        , m_op_count(m_opcodes.size())
        , m_shallowest_read_branch_by_op_index(m_op_count)
        , m_pos_in_dependent_sym_stack_by_op_index(
              m_op_count, DependencyTreeTracker::end_pos())
        , m_pos_in_exec_scope_stack_by_op_index(m_op_count,
                                                FunctionTreeTracker::end_pos())
    {
        // Initially let all symbols be uniform
        // so we get proper cascading of all dependencies
        // when we feed forward from varying shader globals, output parameters, and connected parameters
    }

    void ensure_writes_with_more_conditions_are_masked(
        const Symbol* symbol_to_check, DependencyTreeTracker::Position read_pos)
    {
        // Check if reading a Symbol that was written to from a different
        // dependency lineage than we are reading, if so we need to mark it as requiring masking
        auto lookup = m_write_chronology_by_symbol.find(symbol_to_check);
        if (lookup != m_write_chronology_by_symbol.end()) {
            auto& write_chronology = lookup->second;
            if (!write_chronology.empty()) {
                auto write_end = write_chronology.end();
                // Find common ancestor (ca) between the read position and the write pos
                // if generation of ca is older than current oldest ca for write instruction, record it
                for (auto write_iter = write_chronology.begin();
                     write_iter != write_end; ++write_iter) {
                    auto common_ancestor
                        = m_conditional_symbol_stack.common_ancestor_between(
                            read_pos, write_iter->pos_in_tree());

                    if (common_ancestor != write_iter->pos_in_tree()) {
                        // if common ancestor is the write position, then no need to mask
                        // otherwise we know masking will be needed for the instruction
                        op(write_iter->op_num()).requires_masking(true);
                        OSL_DEV_ONLY(
                            std::cout
                            << "read at shallower depth op("
                            << write_iter->op_num() << ").requires_masking()="
                            << op(write_iter->op_num()).requires_masking()
                            << std::endl);

                        auto ancestor = m_conditional_symbol_stack.begin_at(
                            common_ancestor);

                        // shallowest_read_branch is used to identify the end of the dependencies
                        // that must have symbolFeedForwardMap entries added to correctly
                        // propagate Wide values through the symbols.
                        auto& shallowest_read_branch
                            = m_shallowest_read_branch_by_op_index
                                [write_iter->op_num()];
                        if (ancestor.depth() < shallowest_read_branch.depth) {
                            shallowest_read_branch.depth = ancestor.depth();
                            shallowest_read_branch.pos   = ancestor.pos();
                        }
                    }
                }
            }
        }
    }

    bool
    did_last_write_happen_in_different_loop_cycle(const Symbol* symbol_to_check)
    {
        int loop_op_index = m_loop_stack.current_loop_op_index();
        if (loop_op_index != -1) {
            auto lookup = m_write_chronology_by_symbol.find(symbol_to_check);
            if (lookup != m_write_chronology_by_symbol.end()) {
                auto& write_chronology = lookup->second;
                if (!write_chronology.empty()) {
                    auto& write_event = write_chronology.back();
                    if (write_event.loop_op_index() != loop_op_index) {
                        // The previous write to the symbol happened outside this loop
                        // so if any op writes to it later in the loop
                        // could create a cycle that would require a masked write
                        return true;
                    }

                    auto read_pos = m_conditional_symbol_stack.top_pos();
                    auto common_ancestor
                        = m_conditional_symbol_stack.common_ancestor_between(
                            write_event.pos_in_tree(), read_pos);
                    bool read_not_subset_of_previous_write
                        = common_ancestor != write_event.pos_in_tree();
                    if (read_not_subset_of_previous_write) {
                        // The previous write to the symbol happened in a conditional
                        // whose dependent symbol tree path is different not a superset
                        // of the read.
                        // so if any op writes to it later in the loop
                        // could create a cycle that would require a masked write
                        return true;
                    }
                }
            } else {
                // Read with no previous write, potentially indicates uninitialized data
                // so if any op writes to it later in the loop
                // could create a cycle that would require a masked write
                return true;
            }
        }
        return false;
    };

    void mask_if_cyclic_reads_exist(const Symbol* symbol_to_check,
                                    DependencyTreeTracker::Position write_pos,
                                    int write_op_num)
    {
        // Check if writing a Symbol that was read from a different
        // dependency lineage than we are writing
        auto& read_chronology = m_loop_stack.potentiallyCyclicReadsFor(
            symbol_to_check);
        if (!read_chronology.empty()) {
            auto readEnd = read_chronology.end();
            // If we wrote unmasked, as long as the read as or more restrictive (subset)
            // then it should be fine.  However the reverse is not true, if the read
            // is less restrictive and we wrote unmasked we would be overwriting data
            // lanes the read needed to be maintained.

            // Find common ancestor (ca) between the write position and the read pos
            for (auto read_iter = read_chronology.begin(); read_iter != readEnd;
                 ++read_iter) {
                auto common_ancestor
                    = m_conditional_symbol_stack.common_ancestor_between(
                        write_pos, read_iter->pos_in_tree());
                if (common_ancestor != write_pos) {
                    // if common ancestor is the write position, then no need to mask
                    // otherwise we know masking will be needed for the instruction
                    // as the read was inside a loop and if control flow causes the
                    // loop to iterate it will need to read the symbol that was
                    // written to with a different mask.
                    // NOTE:  if the symbol is read after this write, that is handled
                    // elsewhere to determine if masking in necessary.
                    op(write_op_num).requires_masking(true);
                    OSL_DEV_ONLY(
                        std::cout
                        << "cyclic read of " << symbol_to_check->name().c_str()
                        << " from loop [" << read_iter->loop_op_index()
                        << "] op(" << write_op_num << ").requires_masking()="
                        << op(write_op_num).requires_masking() << std::endl);

                    auto ancestor = m_conditional_symbol_stack.begin_at(
                        common_ancestor);

                    // shallowest_read_branch is used to identify the end of the dependencies
                    // that must have symbolFeedForwardMap entries added to correctly
                    // propagate Wide values through the symbols.
                    auto& shallowest_read_branch
                        = m_shallowest_read_branch_by_op_index[write_op_num];
                    if (ancestor.depth() < shallowest_read_branch.depth) {
                        shallowest_read_branch.depth = ancestor.depth();
                        shallowest_read_branch.pos   = ancestor.pos();
                    }
                }
#ifdef OSL_DEV
                else {
                    std::cout << "rejected cyclic read of "
                              << symbol_to_check->name().c_str()
                              << " from loop [" << read_iter->loop_op_index()
                              << "] for opnum[" << write_op_num << "]"
                              << std::endl;
                }
#endif
            }
        }
    }

    void ensure_writes_with_different_early_out_paths_are_masked(
        const Symbol* symbol_to_check, int op_index,
        FunctionTreeTracker::Position write_scope_pos)
    {
        if (symbol_to_check->renderer_output()) {
            // Any write to a render output must respect masking,
            // also covers a scenario where render output is being initialized by a constant
            // however an earlier init_ops has already exited (meaning those lanes shouldn't get
            // initialized.
            op(op_index).requires_masking(true);
            OSL_DEV_ONLY(std::cout << "render output op(" << op_index
                                   << ").requires_masking()="
                                   << op(op_index).requires_masking()
                                   << std::endl);
        } else {
            // Check if reading a Symbol that was written to from a different
            // dependency lineage than we are reading, if so we need to mark it as requiring masking
            auto lookup = m_write_chronology_by_symbol.find(symbol_to_check);
            if (lookup != m_write_chronology_by_symbol.end()) {
                auto& write_chronology = lookup->second;
                if (!write_chronology.empty()) {
                    auto write_end = write_chronology.end();
                    // If any previous write to the symbol occurred with a different set of early outs
                    // Then the current write needs to be masked
                    for (auto write_iter = write_chronology.begin();
                         write_iter != write_end; ++write_iter) {
                        if (write_iter->is_initial_assignment()) {
                            // The initial assignment of all parameters happens before any instructions are generated?
                            // perhaps there is an ordering issue here for init_ops which could have early outs
                            // although not sure what that would do to execution, certainly returns in init_ops would
                            // bring us back to the m_execution_scope_stack.end_pos()
                            if (write_scope_pos
                                != m_execution_scope_stack.end_pos()) {
                                op(op_index).requires_masking(true);
                                OSL_DEV_ONLY(std::cout
                                             << "early out op(" << op_index
                                             << ").requires_masking()="
                                             << op(op_index).requires_masking()
                                             << std::endl);
                                OSL_DEV_ONLY(
                                    std::cout
                                    << "m_execution_scope_stack.end_pos()()="
                                    << m_execution_scope_stack.end_pos()()
                                    << " write_scope_pos()="
                                    << write_scope_pos()
                                    << " symbol_to_check->name()="
                                    << symbol_to_check->name().c_str()
                                    << std::endl);
                            }
                        } else {
                            if (m_pos_in_exec_scope_stack_by_op_index
                                    [write_iter->op_num()]
                                != write_scope_pos) {
                                op(op_index).requires_masking(true);
                                OSL_DEV_ONLY(std::cout
                                             << "early out op(" << op_index
                                             << ").requires_masking()="
                                             << op(op_index).requires_masking()
                                             << std::endl);
                                OSL_DEV_ONLY(
                                    std::cout
                                    << "m_pos_in_exec_scope_stack_by_op_index[write_iter->op_num()]="
                                    << m_pos_in_exec_scope_stack_by_op_index
                                           [write_iter->op_num()]()
                                    << " write_scope_pos()="
                                    << write_scope_pos()
                                    << " symbol_to_check->name()="
                                    << symbol_to_check->name().c_str()
                                    << std::endl);
                            }
                        }
                    }
                }
            }
        }
    }

    void make_loops_control_flow_depend_on_early_out_conditions()
    {
        // Need change the loop control flow which is dependent upon
        // a conditional.  By making a circular dependency between the this
        // [return|exit|break|continue] operation and the conditionals value,
        // any varying values in the conditional controlling
        // the [return|exit|break|continue] should flow back to the loop control variable,
        // which might need to be varying so allow lanes to terminate the loop independently.
        auto loop_condition = m_loop_stack.current_condition();

        // Now that last loop control condition should exist in our stack of conditions that
        // the current block depends upon, we only need to add dependencies to the loop control
        // to conditionals inside the loop
        OSL_ASSERT(std::find(m_conditional_symbol_stack.begin(),
                             m_conditional_symbol_stack.end(), loop_condition)
                   != m_conditional_symbol_stack.end());
        for (auto cond_iter = m_conditional_symbol_stack.begin();
             *cond_iter != loop_condition; ++cond_iter) {
            auto conditionContinueDependsOn = *cond_iter;
            OSL_DEV_ONLY(std::cout << ">>>Loop Conditional "
                                   << loop_condition->name().c_str()
                                   << " needs to depend on conditional "
                                   << conditionContinueDependsOn->name().c_str()
                                   << std::endl);
            m_symbols_dependent_upon.insert(
                std::make_pair(conditionContinueDependsOn, loop_condition));
        }
    }

    void discover_symbols_between(int beginop, int endop)
    {
        OSL_DEV_ONLY(std::cout << "discover_symbols_between [" << beginop << "-"
                               << endop << "]" << std::endl);
        llvm::SmallVector<Symbol*, 8> symbols_read_by_op;
        llvm::SmallVector<Symbol*, 8> symbols_written_by_op;
        // NOTE: allowing a separate writeMask is to handle condition blocks that are self modifying
        for (int op_index = beginop; op_index < endop; ++op_index) {
            Opcode& opcode = m_opcodes[op_index];
            OSL_DEV_ONLY(std::cout << "op(" << op_index
                                   << ")=" << opcode.opname());
            int arg_count = opcode.nargs();

            // Separate readArgs and WriteArgs
            symbols_read_by_op.resize(arg_count);
            int read_count = 0;
            symbols_written_by_op.resize(arg_count);
            int write_count = 0;
            for (int arg_index = 0; arg_index < arg_count; ++arg_index) {
                auto sym = opargsym(opcode, arg_index);
                if (opcode.argwrite(arg_index)) {
                    OSL_DEV_ONLY(std::cout << " write to ");
                    symbols_written_by_op[write_count++] = sym;
                }
                if (opcode.argread(arg_index)) {
                    symbols_read_by_op[read_count++] = sym;
                    OSL_DEV_ONLY(std::cout << " read from ");
                }
                OSL_DEV_ONLY(std::cout << " " << sym->name());

                OSL_DEV_ONLY(std::cout << " discovery " << sym->name()
                                       << std::endl);
            }
            OSL_DEV_ONLY(std::cout << std::endl);

            // Build up m_symbols_dependent_upon
            // between arguments read to arguments written
            for (int read_index = 0; read_index < read_count; ++read_index) {
                auto read_sym = symbols_read_by_op[read_index];

                if (did_last_write_happen_in_different_loop_cycle(read_sym)) {
                    m_loop_stack.track_potential_cyclic_read(
                        read_sym, m_conditional_symbol_stack.top_pos(),
                        op_index);
                }

                // Some operations can accept a varying input but always return
                // a uniform result.
                // For this set of operations, we skip adding it to the symbolFeedForwardMap
                // as we don't want varying to propagate past them to their return types.
                if (!are_op_results_always_implicitly_uniform(opcode.opname())) {
                    for (int write_index = 0; write_index < write_count;
                         ++write_index) {
                        auto symbolWrittenTo
                            = symbols_written_by_op[write_index];
                        // Skip self dependencies
                        if (symbolWrittenTo != read_sym) {
                            m_symbols_dependent_upon.insert(
                                std::make_pair(read_sym, symbolWrittenTo));
                        }
                    }
                }
                if (write_count == 0) {
                    // Some operations have only side effects and no return value
                    // We still want to track them so they can trigger transition
                    // from uniform to varying if they are a shader global that is varying
                    m_symbols_dependent_upon.insert(
                        std::make_pair(read_sym, nullptr));
                }

                ensure_writes_with_more_conditions_are_masked(
                    read_sym, m_conditional_symbol_stack.top_pos());
            }

            // Process arguments written to to handle early outs, cyclical reads,
            // and record a WriteEvent to a potentially unmasked operations remembering
            // the exact stack of conditional dependencies at that point.
            // When we processed reads (above), we ensured that any reads occuring at a higher  that on the stoCheck for
            for (int write_index = 0; write_index < write_count;
                 ++write_index) {
                const Symbol* symbolWrittenTo
                    = symbols_written_by_op[write_index];

                ensure_writes_with_different_early_out_paths_are_masked(
                    symbolWrittenTo, op_index,
                    m_execution_scope_stack.top_pos());
                mask_if_cyclic_reads_exist(symbolWrittenTo,
                                           m_conditional_symbol_stack.top_pos(),
                                           op_index);

                m_write_chronology_by_symbol[symbolWrittenTo].push_back(
                    WriteEvent(m_conditional_symbol_stack.top_pos(), op_index,
                               m_loop_stack.current_loop_op_index()));
            }

            if (is_op_result_always_logically_boolean(opcode.opname())) {
                // Expect bool result as the 0th argument
                auto boolSymbolWrittenTo = symbols_written_by_op[0];
                m_symbols_logically_bool.insert(boolSymbolWrittenTo);
            } else {
                // To handle a symbol written to by a logical boolean,
                // but later is modified to be an integer, we will
                // remember all symbols that are can NOT be logically boolean
                // we will exclude these from m_symbols_logically_bool
                // at when setting symbol's forced_llvm_bool the end
                for (int write_index = 0; write_index < write_count;
                     ++write_index) {
                    auto symbolWrittenTo = symbols_written_by_op[write_index];
                    m_symbols_disqualified_from_bool.insert(symbolWrittenTo);
                }
            }

            // Add dependencies for operations that implicitly read global variables
            // Those global variables might be varying, and would need their results
            // to be varying
            if (are_op_results_always_implicitly_varying(opcode.opname())) {
                for (int write_index = 0; write_index < write_count;
                     ++write_index) {
                    auto symbolWrittenTo = symbols_written_by_op[write_index];
                    m_symbols_written_to_by_implicitly_varying_ops.insert(
                        symbolWrittenTo);
                }
            }

            // Special case for op_getattribute which normaly would be
            // implicitly varying unless BatchedRendererServices says it
            // is uniform.
            if (opcode.opname() == Strings::op_getattribute) {
                bool object_lookup = opargsym(opcode, 2)->typespec().is_string()
                                     && (arg_count >= 4);
                int object_slot    = static_cast<int>(object_lookup);
                int attrib_slot    = object_slot + 1;
                Symbol& ObjectName = *opargsym(
                    opcode, object_slot);  // only valid if object_slot is true
                Symbol& Attribute = *opargsym(opcode, attrib_slot);

                bool get_attr_is_uniform = false;
                if (Attribute.is_constant()
                    && (!object_lookup || ObjectName.is_constant())) {
                    ustring attr_name = *(const ustring*)Attribute.data();
                    ustring obj_name;
                    if (object_lookup)
                        obj_name = *(const ustring*)ObjectName.data();

                    // TODO:  Perhaps "is_attribute_uniform" should be moved out of width
                    // specific BatchedRendererServices.
                    // Right here we don't know which width will be used,
                    // so we will just require all widths provide the same answer
                    auto rs8  = m_ba.renderer()->batched(WidthOf<8>());
                    auto rs16 = m_ba.renderer()->batched(WidthOf<16>());
                    if (rs8 || rs16) {
                        get_attr_is_uniform = true;
                        if (rs8) {
                            get_attr_is_uniform
                                &= rs8->is_attribute_uniform(obj_name,
                                                             attr_name);
                        }
                        if (rs16) {
                            get_attr_is_uniform
                                &= rs16->is_attribute_uniform(obj_name,
                                                              attr_name);
                        }
                    }
                }

                if (get_attr_is_uniform) {
                    // Set the analysis_flag on the opcode to remember that
                    // get_attribute is uniform for use during code generation
                    // instead of having to ask the renderer a 2nd time
                    opcode.analysis_flag(true);
                } else {
                    for (int write_index = 0; write_index < write_count;
                         ++write_index) {
                        auto symbolWrittenTo
                            = symbols_written_by_op[write_index];
                        m_symbols_written_to_by_implicitly_varying_ops.insert(
                            symbolWrittenTo);
                    }
                }
            }

            // Test if opcode implicitly depends on any protected shader globals,
            // returns SymbolPtrVec * which we can use to create dependencies
            auto implicit_psgs
                = protected_shader_globals_op_implicitly_depends_on(*inst(),
                                                                    opcode);
            if (implicit_psgs != nullptr) {
                for (auto psg_symbol : *implicit_psgs) {
                    for (int write_index = 0; write_index < write_count;
                         ++write_index) {
                        auto symbolWrittenTo
                            = symbols_written_by_op[write_index];
                        m_symbols_dependent_upon.insert(
                            std::make_pair(psg_symbol, symbolWrittenTo));
                    }
                }
            }

            // Some operations implementations can benefit from knowing the mask
            // to ignore lanes when testing for all lanes being off in internal
            // conditional branches.  So we will force calling of the masked version
            // even though it's not strictly required writing to the result be
            // masked.
            if (does_op_implementation_require_masking(opcode.opname())) {
                // Normally, masking would be required because the result
                // symbol is read outside the conditional scope where it
                // was written.  For implementations requiring masking,
                // we defer marking the Opcode as requires_masking until
                // after we have established dependencies for normal masked
                // ops.  We do this because an implementation's
                // requirement for masking should not create dependencies
                // on the conditional symbols stack of this code block.
                // In other words we don't want the result to become
                // varying just because of the implementation's masking
                // requirement.
                m_deferred_op_indices_to_be_masked.push_back(op_index);
            }


            // Track dependencies between symbols written to in this basic block
            // to the set of symbols the code blocks where dependent upon to be executed
            m_pos_in_dependent_sym_stack_by_op_index[op_index]
                = m_conditional_symbol_stack.top_pos();
            m_pos_in_exec_scope_stack_by_op_index[op_index]
                = m_execution_scope_stack.top_pos();

            // Handle control flow
            if (opcode.jump(0) >= 0) {
                // The operation with a jump depends on reading of the
                // condition symbol.  Use m_conditional_symbol_stack to
                // track the condition for the following basic blocks as
                // the writes within those basic blocks will depend on
                // the uniformity of the values read by this operation.
                auto condition = symbols_read_by_op[0];

                // op must have jumps, therefore have nested code we need to process
                // We need to process these in the same order as the code generator
                // so our "block depth" lines up for symbol lookups
                if (opcode.opname() == Strings::op_if) {
                    m_conditional_symbol_stack.push(condition);

                    // Then block
                    m_execution_scope_stack.push_if_block();
                    OSL_DEV_ONLY(std::cout << " THEN BLOCK BEGIN" << std::endl);
                    discover_symbols_between(op_index + 1, opcode.jump(0));
                    OSL_DEV_ONLY(std::cout << " THEN BLOCK END" << std::endl);
                    m_execution_scope_stack.pop_if_block();

                    OSL_ASSERT(m_conditional_symbol_stack.top() == condition);
                    m_conditional_symbol_stack.pop();

                    // else block
                    // NOTE: we are purposefully pushing the same symbol back onto the
                    // dependency tree, this is necessary so that the else block receives
                    // its own unique position in the the dependency tree that we can
                    // tell is different from the then block
                    m_conditional_symbol_stack.push(condition);

                    m_execution_scope_stack.push_else_block();
                    OSL_DEV_ONLY(std::cout << " ELSE BLOCK BEGIN" << std::endl);
                    discover_symbols_between(opcode.jump(0), opcode.jump(1));
                    OSL_DEV_ONLY(std::cout << " ELSE BLOCK END" << std::endl);
                    m_execution_scope_stack.pop_else_block();

                    OSL_ASSERT(m_conditional_symbol_stack.top() == condition);
                    m_conditional_symbol_stack.pop();

                } else if ((opcode.opname() == Strings::op_for)
                           || (opcode.opname() == Strings::op_while)
                           || (opcode.opname() == Strings::op_dowhile)) {
                    // Init block
                    // NOTE: init block doesn't depend on the for loops conditions and should be exempt
                    OSL_DEV_ONLY(std::cout << " FOR INIT BLOCK BEGIN"
                                           << std::endl);
                    discover_symbols_between(op_index + 1, opcode.jump(0));
                    OSL_DEV_ONLY(std::cout << " FOR INIT BLOCK END"
                                           << std::endl);

                    // Save for use later
                    auto treatConditionalAsBeingReadAt
                        = m_conditional_symbol_stack.top_pos();

                    m_conditional_symbol_stack.push(condition);

                    // Only coding for a single conditional variable
                    OSL_ASSERT(read_count == 1);
                    m_loop_stack.push_loop(op_index, symbols_read_by_op[0]);


                    // Body block
                    OSL_DEV_ONLY(std::cout << " FOR BODY BLOCK BEGIN"
                                           << std::endl);
                    discover_symbols_between(opcode.jump(1), opcode.jump(2));
                    OSL_DEV_ONLY(std::cout << " FOR BODY BLOCK END"
                                           << std::endl);

                    // Step block
                    // Because the number of times the step block is executed depends on
                    // when the loop condition block returns false, that means if
                    // the loop condition block is varying, then so would the condition block
                    OSL_DEV_ONLY(std::cout << " FOR STEP BLOCK BEGIN"
                                           << std::endl);
                    discover_symbols_between(opcode.jump(2), opcode.jump(3));
                    OSL_DEV_ONLY(std::cout << " FOR STEP BLOCK END"
                                           << std::endl);

                    OSL_ASSERT(m_conditional_symbol_stack.top() == condition);
                    m_conditional_symbol_stack.pop();


                    // Condition block
                    // NOTE: Processing condition like it was a do/while
                    // Although the first execution of the condition doesn't depend on the for loops conditions
                    // subsequent executions will depend on it on the previous loop's mask
                    // We are processing the condition block out of order so that
                    // any writes to any symbols it depends on can be marked first
                    m_conditional_symbol_stack.push(condition);

                    OSL_DEV_ONLY(std::cout << " FOR COND BLOCK BEGIN"
                                           << std::endl);
                    discover_symbols_between(opcode.jump(0), opcode.jump(1));
                    OSL_DEV_ONLY(std::cout << " FOR COND BLOCK END"
                                           << std::endl);

                    // Special case for symbols that are conditions
                    // because we will be doing horizontal operations on these
                    // to check if they are all 'false' to be able to stop
                    // executing the loop, we need any writes to the
                    // condition to be masked
                    auto condition = opargsym(opcode, 0);
                    ensure_writes_with_more_conditions_are_masked(
                        condition, treatConditionalAsBeingReadAt);

                    OSL_ASSERT(m_conditional_symbol_stack.top() == condition);
                    m_conditional_symbol_stack.pop();
                    m_loop_stack.pop_loop(op_index);


                } else if (opcode.opname() == Strings::op_functioncall) {
                    // Function call itself operates on the same symbol dependencies
                    // as the current block, there was no conditionals involved
                    OSL_DEV_ONLY(std::cout << " FUNCTION CALL BLOCK BEGIN"
                                           << std::endl);
                    m_execution_scope_stack.push_function_call();
                    discover_symbols_between(op_index + 1, opcode.jump(0));
                    m_execution_scope_stack.pop_function_call();
                    OSL_DEV_ONLY(std::cout << " FUNCTION CALL BLOCK END"
                                           << std::endl);

                } else if (opcode.opname() == Strings::op_functioncall_nr) {
                    // Function call itself operates on the same symbol dependencies
                    // as the current block, there was no conditionals involved
                    OSL_DEV_ONLY(std::cout
                                 << " FUNCTION CALL NO RETURN BLOCK BEGIN"
                                 << std::endl);
                    discover_symbols_between(op_index + 1, opcode.jump(0));
                    OSL_DEV_ONLY(std::cout
                                 << " FUNCTION CALL NO RETURN BLOCK END"
                                 << std::endl);
                } else {
                    OSL_ASSERT(
                        0
                        && "Unhandled OSL instruction which contains jumps, note this uniform detection code needs to walk the code blocks identical to build_llvm_code");
                }
            }

            // Handle control flow early outs
            if (opcode.opname() == Strings::op_return) {
                // All operations after this point will also depend on the conditional symbols
                // involved in reaching the return statement.
                // We can lock down the current dependencies to not be removed by
                // scopes until the end of a function
                // NOTE: currently could be overly conservative as I believe this
                // will cause the else block to be locked after a then block with a return.
                m_execution_scope_stack.process_return(
                    m_conditional_symbol_stack.top_pos());

                // The return will need change the loop control flow which is dependent upon
                // a conditional.  By making a circular dependency between the return operation
                // and the conditionals value, any varying values in the conditional controlling
                // the return should flow back to the loop control variable, which might need to
                // be varying so allow lanes to terminate the loop independently
                if (m_loop_stack.is_inside_loop()) {
                    make_loops_control_flow_depend_on_early_out_conditions();
                }
            }
            if (opcode.opname() == Strings::op_exit) {
                // All operations after this point will also depend on the conditional symbols
                // involved in reaching the exit statement.
                // We can lock down the current dependencies to not be removed by
                // scopes until the end of a function
                m_execution_scope_stack.process_exit(
                    m_conditional_symbol_stack.top_pos());

                // The exit will need change the loop control flow which is dependent upon
                // a conditional.  By making a circular dependency between the exit operation
                // and the conditionals value, any varying values in the conditional controlling
                // the exit should flow back to the loop control variable, which might need to
                // be varying so allow lanes to terminate the loop independently
                if (m_loop_stack.is_inside_loop()) {
                    make_loops_control_flow_depend_on_early_out_conditions();
                }
            }
            if (opcode.opname() == Strings::op_break) {
                // All operations in the loop after this point will also depend on
                // the conditional symbols involved in reaching the exit statement.
                // This is automatically handled by the ESTABLISH DEPENDENCIES FOR MASKED INSTRUCTIONS
                // as long as we correctly identified masked instructions, the fixup will
                // hookup dependencies of the conditional stack to that instruction which will
                // allow it to become varying if any of the loop conditionals are varying,
                // and by calling make_loops_control_flow_depend_on_early_out_conditions, if the break was varying
                // so will the loop control
                make_loops_control_flow_depend_on_early_out_conditions();
            }
            if (opcode.opname() == Strings::op_continue) {
                // Track which loops have continue, to minimize code generation which will
                // need to allocate a slot to store the continue mask
                int loop_op_index = m_loop_stack.current_loop_op_index();
                OSL_ASSERT(loop_op_index != -1);
                // Set the analysis_flag of the loop condition operation
                // to identify that a continue exists inside the loop.
                // This will be used during code generation
                op(loop_op_index).analysis_flag(true);

                // All operations in the loop after this point will also depend on
                // the conditional symbols involved in reaching the exit statement.
                // This is automatically handled by the ESTABLISH DEPENDENCIES FOR MASKED INSTRUCTIONS
                // as long as we correctly identified masked instructions, the fixup will
                // hookup dependencies of the conditional stack to that instruction which will
                // allow it to become varying if any of the loop conditionals are varying,
                // and by calling make_loops_control_flow_depend_on_early_out_conditions, if the continue was varying
                // so will the loop control
                make_loops_control_flow_depend_on_early_out_conditions();
            }

            // If the op we coded jumps around, skip past its recursive block
            // executions.
            int next = opcode.farthest_jump();
            if (next >= 0)
                op_index = next - 1;
        }
    };

    void recursively_mark_varying(Symbol* symbol_to_be_varying,
                                  bool force = false)
    {
        bool previously_was_uniform = symbol_to_be_varying->is_uniform();
        if (previously_was_uniform | force) {
            symbol_to_be_varying->make_varying();
            auto range = m_symbols_dependent_upon.equal_range(
                symbol_to_be_varying);
            auto iter = range.first;
            for (; iter != range.second; ++iter) {
                auto dependent_symbol = iter->second;
                // Some symbols read for operations with only side effects and
                // who do not write to another symbol, eg. printf(...)
                if (dependent_symbol != nullptr) {
                    recursively_mark_varying(dependent_symbol);
                }
            };
        }
    };

    void discover_init_symbols()
    {
        // NOTE:  The order symbols are discovered should match the flow
        // of build_llvm_code calls coming from build_llvm_instance
        // And build_llvm_code is called indirectly through llvm_assign_initial_value.
        for (auto&& s : inst()->symbols()) {
            // Skip constants -- we always inline scalar constants, and for
            // array constants we will just use the pointers to the copy of
            // the constant that belongs to the instance.
            if (s.symtype() == SymTypeConst)
                continue;
            // Skip structure placeholders
            if (s.typespec().is_structure())
                continue;
            // Set initial value for constants, closures, and strings that are
            // not parameters.
            if (s.symtype() != SymTypeParam && s.symtype() != SymTypeOutputParam
                && s.symtype() != SymTypeGlobal
                && (s.is_constant() || s.typespec().is_closure_based()
                    || s.typespec().is_string_based()
                    || ((s.symtype() == SymTypeLocal
                         || s.symtype() == SymTypeTemp)
                        && m_ba.shadingsys().debug_uninit()))) {
                if (s.has_init_ops() && s.valuesource() == Symbol::DefaultVal) {
                    // Handle init ops.
                    discover_symbols_between(s.initbegin(), s.initend());
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
                && !s.connected_down() && m_ba.shadingsys().lazy_userdata())
                continue;
            // Set initial value for params (may contain init ops)
            if (s.has_init_ops() && s.valuesource() == Symbol::DefaultVal) {
                // Handle init ops.
                discover_symbols_between(s.initbegin(), s.initend());
            } else {
                // If no init ops exist, must be assigned an constant initial value
                // we must track this write, not because it will need to be masked
                // itself.  But because future writes might happen with a different
                // set of early out which will cause them to masked.  This detection
                // can only happen if we tracked the set of early outs during this
                // initial assignment

                // NOTE: as this is the initial assignment to a parameter
                // there could be no other reads/write to deal with to the symbol
                OSL_ASSERT(m_write_chronology_by_symbol.find(&s)
                           == m_write_chronology_by_symbol.end());

                m_write_chronology_by_symbol[&s].push_back(
                    WriteEvent(m_conditional_symbol_stack.top_pos()));

                // We would check for render outputs and mark it to be masked,
                // but that requires an opindex, and we have no opindex for parameter assignments
                // So we will explicitly check for render outputs at code generation
                // and make their initial assignments masked
            }
        }
    }

    void establish_symbols_forced_llvm_bool()
    {
        for (Symbol* logical_bool_sym : m_symbols_logically_bool) {
            if (m_symbols_disqualified_from_bool.find(logical_bool_sym)
                == m_symbols_disqualified_from_bool.end()) {
                logical_bool_sym->forced_llvm_bool(true);
            }
        }
    }

    void simulate_reading_output_params()
    {
        // Now that all of the instructions have been discovered, we need to
        // make sure any writes to the output parameters that happened at
        // with more dependencies are masked.  As there may be
        // no actual instruction that reads the output variables at the
        // outermost scope that would normally run
        // ensure_writes_with_more_conditions_are_masked, will now simulate a read
        // at the outermost scope
        FOREACH_PARAM(Symbol & s, inst())
        {
            // Skip structure placeholders
            if (s.typespec().is_structure())
                continue;
            // Skip if it's never read and isn't connected
            if (!s.everread() && !s.connected_down() && !s.connected()
                && !s.renderer_output())
                continue;
            if (s.symtype() == SymTypeOutputParam) {
                ensure_writes_with_more_conditions_are_masked(
                    &s, m_conditional_symbol_stack.end_pos());
            }
        }
    }

    void establish_dependencies_for_masked_ops()
    {
        // At this point we should be done figuring out which instructions require masking
        // So those instructions will be dependent on the mask and that mask was
        // dependent on the symbols used in the conditionals that produced it as well
        // as the previous mask on the stack
        // So we need to setup those dependencies, so lets walk through all
        // of the masked instructions and add entries to the m_symbols_dependent_upon
        OSL_DEV_ONLY(std::cout
                     << "ESTABLISH DEPENDENCIES FOR MASKED INSTRUCTIONS"
                     << std::endl);
        for (int op_index = 0; op_index < m_op_count; ++op_index) {
            if (op(op_index).requires_masking()) {
                OSL_DEV_ONLY(std::cout << "masking required for op(" << op_index
                                       << ")" << std::endl);
                {
                    auto begin_dep_iter = m_conditional_symbol_stack.begin_at(
                        m_pos_in_dependent_sym_stack_by_op_index[op_index]);
                    auto end_dep_iter = m_conditional_symbol_stack.begin_at(
                        m_shallowest_read_branch_by_op_index[op_index].pos);

                    const Opcode& opcode = m_opcodes[op_index];
                    int arg_count        = opcode.nargs();
                    for (int arg_index = 0; arg_index < arg_count;
                         ++arg_index) {
                        if (opcode.argwrite(arg_index)) {
                            auto sym_written_to = opargsym(opcode, arg_index);
#ifdef OSL_DEV
                            std::cout << "Symbol written to "
                                      << sym_written_to->name().c_str()
                                      << std::endl;
                            std::cout << "begin_dep_iter "
                                      << begin_dep_iter.pos()() << std::endl;
                            std::cout << "end_dep_iter " << end_dep_iter.pos()()
                                      << std::endl;
#endif
                            for (auto iter = begin_dep_iter;
                                 iter != end_dep_iter; ++iter) {
                                auto sym_mask_depends_on = *iter;
                                // Skip self dependencies
                                if (sym_written_to != sym_mask_depends_on) {
                                    OSL_DEV_ONLY(
                                        std::cout
                                        << "Mapping "
                                        << sym_mask_depends_on->name().c_str()
                                        << std::endl);
                                    m_symbols_dependent_upon.insert(
                                        std::make_pair(sym_mask_depends_on,
                                                       sym_written_to));
                                }
                            }
                        }
                    }
                }

                auto end_of_early_outs = m_execution_scope_stack.end();
                for (auto earlyOutIter = m_execution_scope_stack.begin_at(
                         m_pos_in_exec_scope_stack_by_op_index[op_index]);
                     earlyOutIter != end_of_early_outs; ++earlyOutIter) {
#ifdef OSL_DEV
                    OSL_DEV_ONLY(std::cout
                                 << ">>>>affected_by_a_return op_index "
                                 << op_index << std::endl);
#endif
                    const auto& early_out = *earlyOutIter;
                    auto begin_dep_iter   = m_conditional_symbol_stack.begin_at(
                        early_out.dtt_pos);
                    auto end_dep_iter = m_conditional_symbol_stack.end();

                    const Opcode& opcode = m_opcodes[op_index];
                    int arg_count        = opcode.nargs();
                    for (int arg_index = 0; arg_index < arg_count;
                         ++arg_index) {
                        if (opcode.argwrite(arg_index)) {
                            auto sym_written_to = opargsym(opcode, arg_index);
#ifdef OSL_DEV
                            std::cout << "Symbol written to "
                                      << sym_written_to->name().c_str()
                                      << std::endl;
                            std::cout << "begin_dep_iter "
                                      << begin_dep_iter.pos()() << std::endl;
                            std::cout << "end_dep_iter " << end_dep_iter.pos()()
                                      << std::endl;
#endif
                            for (auto iter = begin_dep_iter;
                                 iter != end_dep_iter; ++iter) {
                                auto sym_mask_depends_on = *iter;
                                // Skip self dependencies
                                if (sym_written_to != sym_mask_depends_on) {
                                    OSL_DEV_ONLY(
                                        std::cout
                                        << "Mapping "
                                        << sym_mask_depends_on->name().c_str()
                                        << std::endl);
                                    m_symbols_dependent_upon.insert(
                                        std::make_pair(sym_mask_depends_on,
                                                       sym_written_to));
                                }
                            }
                        }
                    }
                }
            }
        }
        OSL_DEV_ONLY(std::cout
                     << "END ESTABLISH DEPENDENCIES FOR MASKED INSTRUCTIONS"
                     << std::endl);
    }

    void push_varying_of_shader_globals()
    {
        for (auto&& s : inst()->symbols()) {
            if (s.symtype() == SymTypeGlobal) {
                // TODO: now that symbol has is_uniform()
                // maybe we can just use that (if it were set when the symbol
                // was created).
                if (!is_shader_global_uniform_by_name(s.name())) {
                    // globals may of been marked varying
                    // by a previous layer's analysis,
                    // so force their dependents to get marked
                    recursively_mark_varying(&s, true /*force*/);
                }
            }
        }

        // Handled protected shader global members separately
        for (auto sym_ptr : psg().varying_symbols) {
            // protected shader globals are already marked varying,
            // so force their dependents to get marked
            recursively_mark_varying(sym_ptr, true /*force*/);
        }
    }

    void make_renderer_outputs_varying()
    {
        // Mark all output parameters as varying to catch
        // output parameters written to by uniform variables,
        // as nothing would have made them varying, however as
        // we write directly into wide data, we need to mark it
        // as varying so that the code generation will promote the uniform value
        // to varying before writing
        FOREACH_PARAM(Symbol & s, inst())
        {
            if (s.symtype() == SymTypeOutputParam) {
                // We should only have to do this for outputs that will be pulled by the
                // renderer
                if (s.renderer_output()) {
                    recursively_mark_varying(&s);
                }
            }
        }
    }

    void push_varying_of_upstream_connections()
    {
        OSL_DEV_ONLY(std::cout << "connections to layer begin" << std::endl);
        // Check if any upstream connections are to varying symbols
        // and mark the destination parameters in this layer as varying
        // Discovery goes in order of layers, any upstream symbols
        // should already be discovered and uniformity marked correctly
        {
            ShaderInstance* child = inst();
            int connection_count  = child->nconnections();
            for (int c = 0; c < connection_count; ++c) {
                const Connection& con(child->connection(c));

                ShaderInstance* parent = m_ba.group()[con.srclayer];

                Symbol* srcsym(parent->symbol(con.src.param));
                Symbol* dstsym(child->symbol(con.dst.param));
                // Earlier layers should already be discovered and uniformity mapped to
                // all symbols.  If source symbol is varying,
                // then the dest must be made varying as well
                if (srcsym->is_varying()) {
                    OSL_DEV_ONLY(std::cout
                                 << "symbol " << srcsym->name().c_str()
                                 << " from layer " << con.srclayer
                                 << " is varying and connected to symbol "
                                 << dstsym->name().c_str() << std::endl);
                    recursively_mark_varying(dstsym);
                }
            }
        }
        OSL_DEV_ONLY(std::cout << "connections to layer end" << std::endl);
    }

    void push_varying_of_implicitly_varying_ops()
    {
        OSL_DEV_ONLY(std::cout << "symbolsWrittenToByImplicitlyVaryingOps begin"
                               << std::endl);
        for (auto s : m_symbols_written_to_by_implicitly_varying_ops) {
            OSL_DEV_ONLY(std::cout << s->name() << std::endl);
            recursively_mark_varying(s);
        }
        OSL_DEV_ONLY(std::cout << "symbolsWrittenToByImplicitlyVaryingOps end"
                               << std::endl);
    }

    void process_deferred_masking()
    {
        // Should only be called after establish_dependencies_for_masked_ops has been called
        for (int op_index : m_deferred_op_indices_to_be_masked) {
            op(op_index).requires_masking(true);
        }
    }
};

}  // namespace

BatchedAnalysis::BatchedAnalysis(ShadingSystemImpl& shadingsys,
                                 ShaderGroup& group)
    : m_shadingsys(shadingsys), m_group(group)
{
}

void
BatchedAnalysis::analyze_layer(ShaderInstance* inst)
{
    OSL_DEV_ONLY(std::cout << "start analyze_layer of layer name \""
                           << inst->layername() << "\"" << std::endl);

    OSL_ASSERT(!is_shader_global_uniform_by_name(Strings::shader2common));
    OSL_ASSERT(!is_shader_global_uniform_by_name(Strings::object2common));
    OSL_ASSERT(!is_shader_global_uniform_by_name(Strings::time));

    Analyzer analyzer(*this, inst);

    analyzer.discover_init_symbols();
    analyzer.discover_symbols_between(inst->maincodebegin(),
                                      inst->maincodeend());

    analyzer.establish_symbols_forced_llvm_bool();

    analyzer.simulate_reading_output_params();
    analyzer.establish_dependencies_for_masked_ops();

    OSL_DEV_ONLY(std::cout << "About to find which symbols need to be varying()"
                           << std::endl);
    analyzer.push_varying_of_shader_globals();
    analyzer.make_renderer_outputs_varying();
    analyzer.push_varying_of_upstream_connections();
    analyzer.push_varying_of_implicitly_varying_ops();

    analyzer.process_deferred_masking();
#ifdef OSL_DEV
    dump_symbol_uniformity(inst);
    dump_layer(inst);
#endif
}

void
BatchedAnalysis::dump_symbol_uniformity(ShaderInstance* inst)
{
    {
        std::cout << "Emit Symbol uniformity" << std::endl;

        FOREACH_SYM(Symbol & s, inst)
        {
            std::cout << "--->" << &s << " " << s.name() << " is "
                      << (s.is_uniform() ? "UNIFORM" : "VARYING") << std::endl;
        }
        std::cout << std::flush;
        std::cout << "done with Symbol uniformity" << std::endl;
    }
}

void
BatchedAnalysis::dump_layer(ShaderInstance* inst)
{
    const OpcodeVec& opcodes = inst->ops();
    {
        std::cout << "Emit masking requirements" << std::endl;

        int opCount = opcodes.size();
        for (int op_index = 0; op_index < opCount; ++op_index) {
            const Opcode& opcode = opcodes[op_index];
            if (opcode.requires_masking()) {
                std::cout << "---> inst#" << op_index
                          << " op=" << opcode.opname() << " requires MASKING"
                          << std::endl;
            }
        }
        std::cout << std::flush;
        std::cout << "done with masking requirements" << std::endl;
    }

    {
        std::cout << "Emit analysis_flag for getattribute" << std::endl;

        int opCount = opcodes.size();
        for (int op_index = 0; op_index < opCount; ++op_index) {
            const Opcode& opcode = opcodes[op_index];
            if (opcode.analysis_flag()
                && opcode.opname() == Strings::op_getattribute) {
                std::cout << "---> inst#" << op_index
                          << " op=" << opcode.opname()
                          << " is UNIFORM get_attribute" << std::endl;
            }
        }

        std::cout << std::flush;
        std::cout << "done with analysis_flag for getattribute" << std::endl;
    }

    {
        std::cout << "Emit analysis_flag for loops with continue" << std::endl;
        int opCount = opcodes.size();
        for (int op_index = 0; op_index < opCount; ++op_index) {
            const Opcode& opcode = opcodes[op_index];
            bool isControlFlowOp = (opcode.opname() == Strings::op_for)
                                   || (opcode.opname() == Strings::op_while)
                                   || (opcode.opname() == Strings::op_dowhile);
            if (opcode.analysis_flag() && isControlFlowOp) {
                std::cout << "---> inst#" << op_index
                          << " op=" << opcode.opname()
                          << " is loop with continue" << std::endl;
            }
        }
        std::cout << std::flush;
        std::cout << "done with analysis_flag for loops with continue"
                  << std::endl;
    }

    {
        std::cout << "Emit Symbols forced to llvm bool" << std::endl;

        FOREACH_SYM(Symbol & s, inst)
        {
            if (s.forced_llvm_bool()) {
                std::cout << "--->" << &s << " " << s.name()
                          << " is forced_llvm_bool" << std::endl;
            }
        }
        std::cout << std::flush;
        std::cout << "done with Symbols forced to llvm bool" << std::endl;
    }
}

};  // namespace pvt
OSL_NAMESPACE_EXIT
