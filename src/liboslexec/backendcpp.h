// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <map>
#include <vector>

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;


OSL_NAMESPACE_BEGIN

namespace pvt {  // OSL::pvt


/// OSOProcessor that generates C++ equivalent to the shader network,
/// with main entry function call signature identical to what the JIT
/// produces, so it is exactly substitutable for the JIT results.
class BackendCpp final : public OSOProcessorBase {
public:
    BackendCpp(ShadingSystemImpl& shadingsys, ShaderGroup& group,
               ShadingContext* context);

    virtual ~BackendCpp();

    virtual void run();

    void build_cpp_code(int opbegin, int opend, bool do_indent_block = true);

    /// Write formatted text to the output stream
    template<typename... Args>
    inline void outputfmt(const char* fmt, Args&&... args)
    {
        m_out << fmtformat(fmt, std::forward<Args>(args)...);
    }

    /// Write a full indented line (prepends indentstr(), appends '\n')
    template<typename... Args>
    inline void outputfmtln(const char* fmt, Args&&... args)
    {
        m_out << m_indentview << fmtformat(fmt, std::forward<Args>(args)...)
              << "\n";
    }

    void indent(int delta);
    void increment_indent() { indent(4); }
    void decrement_indent() { indent(-4); }
    string_view indentstr() const { return m_indentview; }

    /// Retrieve the output
    std::string str() const { return m_out.str(); }
    std::ostream& outstream() { return m_out; }

    void generate_groupdata_struct();
    void generate_layer_func(int layer);
    void generate_group_entry();

    // Compact run-flag index for a layer (over used layers only), or -1 if the
    // layer is unused. Populated by run() before any layer/entry emission and
    // shared by generate_layer_func, generate_group_entry, and cpp_gen_useparam.
    int layer_remap(int layer) const { return m_layer_remap[layer]; }

    // Invoke cpp_compiler() to compile cpp_path to a DSO at dso_path.
    // Returns true on success; on failure calls errorfmt() and returns false.
    bool compile_to_dso(const std::string& cpp_path,
                        const std::string& dso_path);

    // Load the DSO at dso_path, verify its OSL_CPP_ABI_VERSION matches the
    // runtime, resolve the group entry symbol, and store both the handle and
    // the entry function pointer on the ShaderGroup. Returns true on success;
    // on failure calls errorfmt(), closes any open handle, and returns false.
    bool load_dso(const std::string& dso_path);

    virtual std::string lang_type_name(TypeDesc type);
    virtual std::string lang_sym_type_name(const Symbol& sym);
    virtual std::string lang_preamble();
    virtual std::string lang_function_qualifier();
    virtual std::string lang_linkage_prefix();
    virtual std::string lang_file_extension();
    virtual std::string lang_ptr_syntax();

    std::string cpp_var_declaration(const Symbol& sym);

    // Return the C++ value string for a symbol: for SymTypeConst, returns the
    // literal value (floats with 'f' suffix); otherwise returns cpp_safe_name().
    std::string cpp_value_str(const Symbol& sym);

    // Materialize a const scalar arg so its address can be taken, returning
    // the `(void*)&...` expression.
    std::string cpp_void_ptr_arg(const Symbol& sym, const std::string& tmpname);

    // Return a "double quoted" string literal, with special characters
    // escaped.
    std::string quoted_string(string_view s) const;

    // Return the in-language representation of a quoted string literal. For
    // example, in C++ meant for OSL JIT, this might be
    // `OSL::ustring("blah").hash()`.
    std::string cpp_string_literal_rep(string_view s) const;

    // debug_uninit helpers
    void cpp_uninit_marker_init(const Symbol& s);
    void cpp_generate_debug_uninit(int opnum);

    // Return true if the op range [opbegin, opend) contains a 'continue' op
    // at this loop nesting level (skipping over nested loop bodies).
    bool body_has_continue(int opbegin, int opend);

    // Return true if the op range [opbegin, opend) contains a 'return' op that
    // belongs to THIS inlined function (skipping over nested function bodies).
    // Used to decide whether a functioncall body needs a goto-label so 'return'
    // jumps to the end of the inlined body instead of out of the layer function.
    bool body_has_return(int opbegin, int opend);

    // Loop context stack. Each entry is the goto-target for 'continue' in the
    // innermost loop. An empty string means emit the natural 'continue;' keyword.
    // 'break' always emits the natural 'break;' keyword.
    int new_loop_label_id() { return m_loop_label_counter++; }
    void push_loop_context(std::string cont_tgt)
    {
        m_loop_ctx.push_back(std::move(cont_tgt));
    }
    void pop_loop_context() { m_loop_ctx.pop_back(); }
    const std::string& loop_cont_target() const { return m_loop_ctx.back(); }

    // Function context stack for inlined functioncall bodies. Each entry is the
    // goto-target a 'return' jumps to (the end of the inlined body); an empty
    // string means the body has no 'return' and no label was emitted. An 'exit'
    // op always emits a natural 'return;' to leave the whole layer function.
    void push_func_context(std::string ret_tgt)
    {
        m_func_ctx.push_back(std::move(ret_tgt));
    }
    void pop_func_context() { m_func_ctx.pop_back(); }
    bool inside_function() const { return !m_func_ctx.empty(); }
    const std::string& func_return_target() const { return m_func_ctx.back(); }

    // True when sym carries derivs, which in C++ meand it's declared as an
    // `OSL::Dual2` in generated code).
    bool sym_carries_derivs(const Symbol& s) const;

    // Return the C++ expression for component (a=element, c=sub-component) of
    // sym for use in the printf value buffer.  For symbols with real storage
    // this is an addressable lvalue.  Scalar/aggregate constants are inlined
    // as literals (they are never declared as variables), so for those we
    // return the literal value and set *needs_temp so the caller materializes
    // a temporary to take its address.  String/array constants have backing
    // storage and stay addressable.
    std::string printf_arg_expr(const Symbol& sym, int a, int c,
                                bool* needs_temp) const;

    // A coordinate-system / colorspace name as a ustringhash_pod, as expected
    // by the osl_* transform calls.  String constants are emitted as a
    // uint64_t hash (already a pod); string variables are OSL::ustringhash
    // and need .hash().
    std::string cpp_spacename_pod(const Symbol& s);

    // A float component value for a constructor, stripping a Dual2 scalar to
    // its .val().
    std::string cpp_float_val(const Symbol& s);

    // Scalar arg passed by value to a runtime function: strip a Dual2 to its
    // value.
    std::string cpp_scalar_val(const Symbol& s);

    // Build the constructor expression for a triple Result from three float
    // component symbols. When Result carries derivatives and any component is
    // itself a Dual2, assemble per-component val/dx/dy; otherwise emit a plain
    // 3-arg (optionally Dual2-wrapped) ctor.
    std::string cpp_triple_ctor(const Symbol& R, const Symbol* c0,
                                const Symbol* c1, const Symbol* c2);

    // Emit `R = Matrix44(diag,0,0,0, 0,diag,0,0, ...)`: a diagonal matrix with
    // the scalar `diag` on the diagonal.
    void cpp_emit_matrix_diagonal(const Symbol& R, const std::string& diag);

    // Whole-array copy R = A, handling matched deriv-ness (memcpy of
    // min(dst,src) elements) and mismatched deriv-ness (element-wise copy).
    void cpp_array_copy(const Symbol& R, const Symbol& A);

    // Return the C++ index expression to use for an indexed access, wrapping it
    // in osl_range_check when the shader has range checking enabled and the
    // index is not a provably in-range constant. `symname` is the name shown in
    // the error message (the array/matrix symbol).
    std::string cpp_range_check(const Opcode& op, const Symbol& Index,
                                int length, string_view symname);

private:
    bool cpp_can_treat_param_as_local(const Symbol& sym) const;
    static std::string cpp_struct_element_type(TypeDesc type);
    std::string cpp_const_value_str(const Symbol& sym);
    std::string cpp_const_literal_str(const Symbol& sym);

    // Format a finite float as a C++ float literal with 'f' suffix, using 9
    // significant digits so the emitted constant has the same bits as the JIT's.
    std::string float_lit(float v) const;
    int m_indentlevel        = 0;
    int m_loop_label_counter = 0;
    string_view m_indentview;
    std::ostringstream m_out;
    std::vector<std::string> m_loop_ctx;
    std::vector<std::string> m_func_ctx;
    std::vector<int> m_layer_remap;

    void op_gen_init();
};



};  // namespace pvt
OSL_NAMESPACE_END
