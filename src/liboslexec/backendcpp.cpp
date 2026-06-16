// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <mutex>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/fmath.h>

#include <OSL/encodedtypes.h>

#include "oslexec_pvt.h"

#include "backendcpp.h"

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_BEGIN

namespace pvt {


BackendCpp::BackendCpp(ShadingSystemImpl& shadingsys, ShaderGroup& group,
                       ShadingContext* ctx)
    : OSOProcessorBase(shadingsys, group, ctx)
{
    op_gen_init();
}



BackendCpp::~BackendCpp() {}



bool
BackendCpp::compile_to_dso(const std::string& cpp_path,
                           const std::string& dso_path)
{
    // When OSL_CPP_SKIP_COMPILE=1 the compilation step is suppressed so that
    // a test can pre-place a stub DSO at `dso_path` and exercise load_dso()
    // without running the compiler.  The DSO must already exist.
    const char* skip_env = getenv("OSL_CPP_SKIP_COMPILE");
    if (skip_env && std::string(skip_env) == "1")
        return OIIO::Filesystem::exists(dso_path);

    std::string cmd = fmtformat("\"{}\" {} -o \"{}\" \"{}\" 2>&1",
                                shadingsys().cpp_compiler(),
                                shadingsys().cpp_compiler_flags(), dso_path,
                                cpp_path);
    // Remove any stale DSO so its existence after the command is a reliable
    // success indicator (compilers don't write output on failure).
    OIIO::Filesystem::remove(dso_path);
    std::string output;
    if (!OIIO::Filesystem::read_text_from_command(cmd, output)) {
        shadingsys().errorfmt("BackendCpp: could not launch compiler: {}", cmd);
        return false;
    }
    if (!OIIO::Filesystem::exists(dso_path)) {
        shadingsys().errorfmt("BackendCpp: DSO compilation failed:\n{}",
                              output);
        return false;
    }
    return true;
}



bool
BackendCpp::load_dso(const std::string& dso_path)
{
    OIIO::Plugin::Handle handle = OIIO::Plugin::open(dso_path,
                                                     /*global=*/false);
    if (!handle) {
        shadingsys().errorfmt("BackendCpp: could not load DSO {}: {}", dso_path,
                              OIIO::Plugin::geterror());
        return false;
    }

    // Verify the DSO was generated against a compatible ABI before trusting
    // any of its other symbols.
    using AbiFunc       = int (*)();
    AbiFunc abi_version = reinterpret_cast<AbiFunc>(
        OIIO::Plugin::getsym(handle, "osl_cpp_abi_version",
                             /*report_error=*/false));
    if (!abi_version) {
        shadingsys().errorfmt(
            "BackendCpp: DSO {} is missing the osl_cpp_abi_version symbol",
            dso_path);
        OIIO::Plugin::close(handle);
        return false;
    }
    int dso_abi = abi_version();
    if (dso_abi != OSL_CPP_ABI_VERSION) {
        shadingsys().errorfmt(
            "BackendCpp: DSO {} ABI version {} does not match runtime ABI version {}",
            dso_path, dso_abi, OSL_CPP_ABI_VERSION);
        OIIO::Plugin::close(handle);
        return false;
    }

    // Resolve the group entry point. The symbol name must match the one
    // emitted by generate_group_entry() (raw group name, extern "C").
    std::string entry_name = fmtformat("osl_init_group_{}", group().name());
    RunLLVMGroupFunc entry = reinterpret_cast<RunLLVMGroupFunc>(
        OIIO::Plugin::getsym(handle, entry_name, /*report_error=*/false));
    if (!entry) {
        shadingsys().errorfmt(
            "BackendCpp: DSO {} is missing the group entry symbol {}", dso_path,
            entry_name);
        OIIO::Plugin::close(handle);
        return false;
    }

    group().cpp_dso_handle(handle);
    group().cpp_compiled_version(entry);
    return true;
}



static std::string indent_reservoir(128, ' ');



void
BackendCpp::indent(int delta)
{
    m_indentlevel += delta;
    m_indentview = string_view(indent_reservoir.c_str(),
                               OIIO::clamp(size_t(m_indentlevel), size_t(0),
                                           indent_reservoir.size()));
}



std::string
BackendCpp::lang_preamble()
{
    return "#include \"osl_cpp_runtime.h\"";
}



std::string
BackendCpp::lang_function_qualifier()
{
    return "";
}



std::string
BackendCpp::lang_linkage_prefix()
{
    return "extern \"C\"";
}



std::string
BackendCpp::lang_file_extension()
{
    return ".cpp";
}



std::string
BackendCpp::lang_ptr_syntax()
{
    return "*";
}



std::string
BackendCpp::lang_type_name(TypeDesc type)
{
    TypeDesc scalar = type;
    scalar.arraylen = 0;
    return cpp_struct_element_type(scalar);
}



std::string
BackendCpp::lang_sym_type_name(const Symbol& sym)
{
    std::string str;
    TypeSpec t = sym.typespec();
    if (t.is_closure() || t.is_closure_array()) {
        // A closure value is just a pointer (ClosureColor*).  The array bound,
        // if any, is appended by cpp_var_declaration() in declarator position
        // (`closure_color_t name[N]`), not embedded in the type.
        str = "closure_color_t";
    } else if (t.structure() > 0) {
        StructSpec* ss = t.structspec();
        if (ss)
            str += fmtformat("struct {}", t.structspec()->name());
        else
            str += fmtformat("struct {}", t.structure());
        if (t.is_unsized_array())
            str += "[]";
        else if (t.arraylength() > 0)
            str += fmtformat("[{}]", t.arraylength());
    } else {
        str = lang_type_name(t.simpletype());
        // Derivative-carrying float scalars and triples are promoted to
        // OSL::Dual2<...>.  Dual2<Vec3>/Dual2<Color3> is 36 contiguous bytes
        // (val,dx,dy), matching the osl_*_dv... deriv-triple void* ABI.
        TypeDesc st       = t.simpletype();
        bool deriv_scalar = st.aggregate == TypeDesc::SCALAR
                            && st.basetype == TypeDesc::FLOAT;
        bool deriv_triple = st.aggregate == TypeDesc::VEC3
                            && st.basetype == TypeDesc::FLOAT;
        // Deriv-carrying arrays are declared Dual2<elem>[N] (AoS); the array
        // bound is appended by cpp_var_declaration. This matches the GroupData
        // deriv-array fields. Whole-array passes to runtime functions that
        // expect the SoA deriv layout (e.g. spline knots) build a SoA shadow at
        // the call site (see cpp_gen_spline).
        if (sym.has_derivs() && (deriv_scalar || deriv_triple))
            str = fmtformat("OSL::Dual2<{}>", str);
    }
    return str;
}



std::string
BackendCpp::cpp_var_declaration(const Symbol& sym)
{
    const char* qualifier = (sym.symtype() == SymTypeConst) ? "const " : "";
    std::string decl = fmtformat("{}{} {}", qualifier, lang_sym_type_name(sym),
                                 sym.cpp_safe_name());
    // Arrays carry their bound in the declarator — `float arr[4]` — not the
    // type (closure arrays included: `closure_color_t arr[4]`).  Structs still
    // embed their bound in lang_sym_type_name's `struct S[N]` spelling.
    TypeSpec ts = sym.typespec();
    if (ts.is_array() && ts.structure() == 0) {
        int len = ts.is_unsized_array() ? sym.initializers() : ts.arraylength();
        decl += fmtformat("[{}]", len);
    }
    return decl;
}



bool
BackendCpp::cpp_can_treat_param_as_local(const Symbol& sym) const
{
    if (!shadingsys().m_opt_groupdata)
        return false;
    return sym.symtype() == SymTypeOutputParam && !sym.renderer_output()
           && !sym.typespec().is_closure_based() && !sym.connected();
}



std::string
BackendCpp::cpp_struct_element_type(TypeDesc type)
{
    TypeDesc scalar = type;
    scalar.arraylen = 0;
    if (scalar.basetype == TypeDesc::STRING)
        scalar.basetype = TypeDesc::USTRINGHASH;

    if (scalar.aggregate == TypeDesc::SCALAR) {
        switch (scalar.basetype) {
        case TypeDesc::FLOAT: return "float";
        case TypeDesc::INT: return "int";
        case TypeDesc::INT8: return "int8_t";
        case TypeDesc::USTRINGHASH: return "OSL::ustringhash";
        default: break;
        }
    } else if (scalar.aggregate == TypeDesc::VEC3) {
        if (scalar.vecsemantics == TypeDesc::COLOR)
            return "OSL::Color3";
        return "OSL::Vec3";
    } else if (scalar.aggregate == TypeDesc::MATRIX44) {
        return "OSL::Matrix44";
    }
    return std::string(scalar.c_str());
}



void
BackendCpp::generate_groupdata_struct()
{
    int nlayers         = group().nlayers();
    int num_used_layers = 0;
    for (int i = 0; i < nlayers; ++i)
        if (!group()[i]->unused())
            ++num_used_layers;

    outputfmtln("struct GroupData {{");
    increment_indent();

    // Field 0: layer run flags rounded up to 32-bit boundary
    int sz = (num_used_layers + 3) & (~3);
    outputfmtln("bool layer_runflags[{}];", sz);

    // Userdata init flags and value fields
    int nuserdata = (int)group().m_userdata_names.size();
    if (nuserdata) {
        int ud_sz = (nuserdata + 3) & (~3);
        outputfmtln("int8_t userdata_init_flags[{}];", ud_sz);
        for (int i = 0; i < nuserdata; ++i) {
            TypeDesc type = group().m_userdata_types[i];
            // Mirror the JIT's llvm_type_groupdata userdata sizing exactly: a
            // float-based userdata field ALWAYS reserves room for derivatives
            // (numelements*3), whether or not this param uses them; non-float
            // reserves numelements. Sizing by m_userdata_derivs instead would
            // shrink the field and shift every later GroupData param's offset,
            // so get_symbol() (which reads at the JIT's dataoffset) would read
            // the wrong location — e.g. userdata-passthrough's Cd output.
            int total       = (type.basetype == TypeDesc::FLOAT)
                                  ? type.numelements() * 3
                                  : type.numelements();
            TypeDesc scalar = type;
            scalar.arraylen = 0;
            outputfmtln("{} userdata{}_{}_[{}];",
                        cpp_struct_element_type(scalar), i,
                        group().m_userdata_names[i], total);
        }
    }

    // Per-layer, per-param fields (those not eligible to be stack-locals)
    for (int layer = 0; layer < nlayers; ++layer) {
        ShaderInstance* linst = group()[layer];
        if (linst->unused())
            continue;
        FOREACH_PARAM(Symbol & sym, linst)
        {
            TypeSpec ts = sym.typespec();
            if (ts.is_structure())
                continue;
            if (cpp_can_treat_param_as_local(sym))
                continue;
            if (ts.is_closure() || ts.is_closure_array()) {
                // A closure value is a pointer; connected closures copy the
                // pointer down through this slot.
                if (ts.is_array()) {
                    if (ts.is_unsized_array()) {
                        outputfmtln(
                            "// UNIMPLEMENTED: unsized closure array param lay{}param_{}",
                            layer, sym.cpp_safe_name());
                    } else {
                        outputfmtln("closure_color_t lay{}param_{}[{}];", layer,
                                    sym.cpp_safe_name(), ts.arraylength());
                    }
                } else {
                    outputfmtln("closure_color_t lay{}param_{};", layer,
                                sym.cpp_safe_name());
                }
                continue;
            }
            if (ts.is_unsized_array()) {
                // Safety guard: the optimizer normally resolves unsized-array
                // param sizes before BackendCpp runs, but guard here to prevent
                // a DASSERT in arraylength() if one ever slips through.
                outputfmtln(
                    "// UNIMPLEMENTED: unsized array param lay{}param_{}",
                    layer, sym.cpp_safe_name());
                continue;
            }
            const bool is_arr       = ts.is_array();
            const int arraylen      = ts.arraylength();
            const bool has_derivs   = sym.has_derivs();
            const std::string elt   = cpp_struct_element_type(ts.simpletype());
            const std::string fname = sym.cpp_safe_name();
            if (has_derivs) {
                if (is_arr)
                    outputfmtln("OSL::Dual2<{}> lay{}param_{}[{}];", elt, layer,
                                fname, arraylen);
                else
                    outputfmtln("OSL::Dual2<{}> lay{}param_{};", elt, layer,
                                fname);
            } else if (is_arr) {
                outputfmtln("{} lay{}param_{}[{}];", elt, layer, fname,
                            arraylen);
            } else {
                outputfmtln("{} lay{}param_{};", elt, layer, fname);
            }
        }
    }

    decrement_indent();
    outputfmt("}};\n\n");
}



// Format a finite float as a C++ float literal with 'f' suffix, ensuring a
// decimal point so the compiler treats it as float (not double).  Uses 9
// significant digits — the minimum that round-trips any IEEE single — so the
// emitted constant has the exact same bits as the JIT's (e.g. M_PI must be
// 3.14159274f, not the lossy 3.14159f that {:g} would produce).
std::string
BackendCpp::float_lit(float v) const
{
    std::string s = fmtformat("{:.9g}", v);
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos)
        s += ".0";
    return s + "f";
}



// Return the literal C++ initializer expression for a constant symbol: a bare
// scalar literal (int, or float with 'f' suffix), an aggregate constructor
// (Color3, Vec3, Matrix44, …), or a brace-enclosed initializer list for an
// array of either.  Used as the initializer when a constant needs a named
// declaration.  (String constants are handled separately at the declaration
// site and never reach here.)
std::string
BackendCpp::cpp_const_literal_str(const Symbol& sym)
{
    TypeDesc td = sym.typespec().simpletype();
    int nagg    = td.aggregate;

    // One scalar-or-aggregate element starting at flat component offset `base`.
    auto one = [&](int base) -> std::string {
        if (td.basetype == TypeDesc::STRING
            || td.basetype == TypeDesc::USTRINGHASH)
            // OSL::ustringhash has an implicit ctor from OSL::ustring.
            return fmtformat("OSL::ustring(\"{}\")",
                             quoted_string(sym.get_string(base)));
        if (nagg == 1)
            return (td.basetype == TypeDesc::FLOAT)
                       ? float_lit(sym.get_float(base))
                       : fmtformat("{}", sym.get_int(base));
        // Aggregate (Color3, Vec3, Matrix44, …): name the element type, not
        // the array type, so strip any array bound before lang_type_name.
        TypeDesc elem = td;
        elem.arraylen = 0;
        std::string s = lang_type_name(elem) + "(";
        for (int c = 0; c < nagg; ++c) {
            if (c > 0)
                s += ", ";
            s += float_lit(sym.get_float(base + c));
        }
        return s + ")";
    };

    if (td.arraylen == 0)
        return one(0);

    // Array constant: brace-enclosed initializer list.
    std::string s = "{ ";
    int nelem     = td.numelements();
    for (int e = 0; e < nelem; ++e) {
        if (e > 0)
            s += ", ";
        s += one(e * nagg);
    }
    return s + " }";
}



// Return the C++ source representation of a constant symbol's value at a use
// site.  Scalars are inlined as literals.  Strings, arrays, and aggregates have
// real backing storage (a named declaration) so their address can be taken when
// passed by void* to osl_* functions — for those, return the variable name.
std::string
BackendCpp::cpp_const_value_str(const Symbol& sym)
{
    TypeDesc td    = sym.typespec().simpletype();
    bool is_scalar = (td.arraylen == 0 && td.aggregate == 1
                      && td.basetype != TypeDesc::STRING
                      && td.basetype != TypeDesc::USTRINGHASH);
    return is_scalar ? cpp_const_literal_str(sym) : sym.cpp_safe_name();
}



std::string
BackendCpp::cpp_value_str(const Symbol& sym)
{
    return (sym.symtype() == SymTypeConst) ? cpp_const_value_str(sym)
                                           : sym.cpp_safe_name();
}



// Scan [opbegin, opend) for a 'continue' op that belongs to THIS loop level
// (not nested inside an inner loop). Used to decide whether a for/dowhile
// loop body requires a step-label goto for correct 'continue' semantics.
bool
BackendCpp::body_has_continue(int opbegin, int opend)
{
    static ustring s_continue("continue");
    static ustring s_for("for");
    static ustring s_while("while");
    static ustring s_dowhile("dowhile");
    for (int i = opbegin; i < opend; ++i) {
        const Opcode& op = inst()->ops()[i];
        if (op.opname() == s_continue)
            return true;
        // Skip nested loops only — their continue belongs to them. A continue
        // inside an 'if' (or other conditional) of THIS loop belongs to this
        // loop, and its body ops are in the linear op stream, so it must NOT be
        // skipped (mirrors body_has_return's nested-functioncall handling).
        if (op.opname() == s_for || op.opname() == s_while
            || op.opname() == s_dowhile) {
            int next = op.farthest_jump();
            if (next >= 0)
                i = next - 1;
        }
    }
    return false;
}



// Scan [opbegin, opend) for a 'return' op that belongs to THIS inlined function
// (not nested inside an inner functioncall). 'return' ops inside loops/ifs of
// this function DO belong to it, so only nested function bodies are skipped.
bool
BackendCpp::body_has_return(int opbegin, int opend)
{
    static ustring s_return("return");
    static ustring s_functioncall("functioncall");
    static ustring s_functioncall_nr("functioncall_nr");
    for (int i = opbegin; i < opend; ++i) {
        const Opcode& op = inst()->ops()[i];
        if (op.opname() == s_return)
            return true;
        // Skip nested function bodies — their returns belong to them.
        if (op.opname() == s_functioncall || op.opname() == s_functioncall_nr) {
            int next = op.farthest_jump();
            if (next >= 0)
                i = next - 1;
        }
    }
    return false;
}



void
BackendCpp::generate_layer_func(int layer)
{
    set_inst(layer);
    if (inst()->unused())
        return;
    find_basic_blocks();

    std::string group_name = group().name().string();
    std::string func_name  = fmtformat("osl_layer_group_{}_name_{}", group_name,
                                       inst()->layername());

    std::string qual = lang_function_qualifier();
    outputfmt("{}{}static void {}(\n", qual.empty() ? "" : qual + " ",
              indentstr(), func_name);
    outputfmt("{}    OSL::ShaderGlobals{} sg, GroupData{} gd,\n", indentstr(),
              lang_ptr_syntax(), lang_ptr_syntax());
    outputfmt("{}    void{} userdata_base, void{} output_base,\n", indentstr(),
              lang_ptr_syntax(), lang_ptr_syntax());
    outputfmt("{}    int shadeindex, void{} interactive_params)\n", indentstr(),
              lang_ptr_syntax());
    outputfmtln("{{");
    increment_indent();

    outputfmtln("// Layer {}: {} (Shader {})", layer, inst()->layername(),
                inst()->shadername());

    // Mark this layer as run at the very start, so an on-demand call from a
    // downstream layer's cpp_gen_useparam (or the group entry) runs it at most
    // once. Mirrors the JIT, where the layer function sets its own run-flag.
    outputfmtln("gd->layer_runflags[{}] = true;", m_layer_remap[layer]);

    // Load params from GroupData (if connected) or initialize with default values.
    FOREACH_PARAM(Symbol & s, inst())
    {
        if (!s.everused())
            continue;
        TypeSpec ts = s.typespec();
        if (ts.is_structure())
            continue;
        // cpp_var_declaration() spells the full declarator including the array
        // bound for array params; lang_sym_type_name alone omits it, which would
        // declare an array param as a scalar (so `name[i]` fails to compile).
        std::string decl = cpp_var_declaration(s);
        // A param runs its init ops only when its value comes from the default
        // (mirrors the JIT's llvm_assign_initial_value: init ops run iff
        // valuesource()==DefaultVal). An instance-overridden param (InstanceVal,
        // e.g. set via --param) loads its literal value instead, and its init
        // ops are skipped. Default-valued init ops are emitted in a deferred
        // pass below, after all locals/temps/globals are declared (init ops
        // reference them) and before the main code.
        bool runs_init_ops = s.has_init_ops()
                             && s.valuesource() == Symbol::DefaultVal;
        if (cpp_can_treat_param_as_local(s)) {
            // Output-only param with no GroupData slot: a plain local. Seed it
            // with its constant/default value (unless default-valued init ops
            // will, in the deferred pass). Without this, an output whose
            // value-setting op was constant-folded away — leaving empty main
            // code — is left uninitialized: e.g. a pure-constant output feeding
            // a component connection at -O2 (connect-components).
            if (runs_init_ops)
                outputfmtln("{};", decl);
            else
                outputfmtln("{} = {};", decl, cpp_const_literal_str(s));
        } else if (s.connected()) {
            // Connected from upstream: declare uninitialized. The value is
            // loaded from GroupData (after running the upstream layer on demand)
            // by cpp_gen_useparam at the point of use, so a run_lazily() upstream
            // executes lazily and in the JIT's order.
            outputfmtln("{};", decl);
        } else if (s.interactive()
                   && group().interactive_param_offset(layer, s.name()) >= 0) {
            // Interactively-adjusted param: its current value lives in the
            // interactive arena (passed as interactive_params), not in
            // GroupData or a baked-in default — so reparam takes effect.
            // The arena holds [val][dx][dy] contiguous at this offset, which
            // matches the local's layout (incl. Dual2 for deriv-carrying
            // params), so a single memcpy of sizeof(local) is correct.
            // Mirrors BackendLLVM::getLLVMSymbolBase's interactive case.
            int off = group().interactive_param_offset(layer, s.name());
            outputfmtln("{};", decl);
            if (ts.is_array())
                outputfmtln(
                    "std::memcpy({}, (char*)interactive_params + {}, sizeof({}));",
                    s.cpp_safe_name(), off, s.cpp_safe_name());
            else
                outputfmtln(
                    "std::memcpy(&{}, (char*)interactive_params + {}, sizeof({}));",
                    s.cpp_safe_name(), off, s.cpp_safe_name());
        } else if (s.interpolated() && !ts.is_closure_based()) {
            // Interpolated (lockgeom=0) param: its value is bound from the
            // renderer's userdata in the deferred pass below (which can fall back
            // to running default init ops). Declare uninitialized here.
            outputfmtln("{};", decl);
        } else if (ts.is_closure_based()) {
            // A closure's only constant value is null. Connected closures took
            // the branch above; an unconnected closure param whose default-valued
            // init ops will run is declared uninitialized (the deferred pass
            // assigns it).
            if (runs_init_ops)
                outputfmtln("{};", decl);
            else
                outputfmtln("{} = nullptr;", decl);
        } else if (runs_init_ops) {
            // Default-valued param with init ops: declare uninitialized; the
            // deferred init-op pass below assigns it.
            outputfmtln("{};", decl);
        } else {
            // Constant default or instance-value override (InstanceVal): the
            // value lives in the symbol's data, so initialize directly with the
            // literal. A param's default/instance value is not a separately
            // declared constant, so it must be spelled as a literal —
            // cpp_const_value_str would return the (here, self-) variable name.
            // cpp_const_literal_str handles scalars, aggregates, strings, and
            // arrays of those.
            outputfmtln("{} = {};", decl, cpp_const_literal_str(s));
        }
    }

    // Declare constants, temps, locals
    FOREACH_SYM(Symbol & s, inst())
    {
        if (!s.everused())
            continue;
        if (s.symtype() == SymTypeConst) {
            // Scalars are inlined at each use via cpp_value_str(). Strings,
            // arrays, and aggregates need a named variable so their address can
            // be taken (aggregates are passed by void* to osl_* functions).
            // Strings: static const so OSL::ustring(...).hash() runs once
            // and the hash is cached for every subsequent shader invocation.
            // Declared as the raw uint64_t hash because osl_* runtime functions
            // take string args by value as ustringhash_pod (an unsigned 64-bit
            // integer); assignment to a ustringhash variable wraps it via
            // ustringhash::from_hash (see cpp_gen_assign).
            TypeDesc td    = s.typespec().simpletype();
            bool is_string = (td.arraylen == 0
                              && (td.basetype == TypeDesc::STRING
                                  || td.basetype == TypeDesc::USTRINGHASH));
            if (is_string) {
                outputfmtln("static const uint64_t {} = {};", s.cpp_safe_name(),
                            cpp_string_literal_rep(s.get_string()));
            } else if (td.arraylen > 0) {
                outputfmtln("{} = {};", cpp_var_declaration(s),
                            cpp_const_literal_str(s));
            } else if (td.aggregate > 1) {
                outputfmtln("{} = {};", cpp_var_declaration(s),
                            cpp_const_literal_str(s));
            }
        } else if (s.symtype() == SymTypeTemp || s.symtype() == SymTypeLocal) {
            outputfmtln("{};", cpp_var_declaration(s));
            // With debug_uninit, fill locals/temps with the uninitialized marker
            // so a read before assignment can be detected.
            if (shadingsys().debug_uninit())
                cpp_uninit_marker_init(s);
        } else if (s.symtype() == SymTypeGlobal) {
            // Load shader global from sg; s.name() matches the ShaderGlobals field.
            // Globals that carry derivatives (scalar u/v/time, triple P/I, ...)
            // are declared as a Dual2 and need it constructed from the base field
            // plus the dx/dy fields: "d" + name + "dx" / "dy".  For triples the
            // dx/dy SG fields are Vec3, matching Dual2<Vec3>(Vec3,Vec3,Vec3).
            if (sym_carries_derivs(s)) {
                std::string sn = s.name().string();
                outputfmtln("{} {}(sg->{}, sg->d{}dx, sg->d{}dy);",
                            lang_sym_type_name(s), s.cpp_safe_name(), sn, sn,
                            sn);
            } else {
                outputfmtln("{} {} = sg->{};", lang_sym_type_name(s),
                            s.cpp_safe_name(), s.name());
            }
        }
    }

    // Run default-valued params' init ops, now that every local/temp/const/
    // global they reference has been declared. Mirrors the JIT's
    // llvm_assign_initial_value: a param runs its init ops only when its value
    // comes from the default (valuesource()==DefaultVal). Instance-overridden
    // (InstanceVal), connected, and interactive params took their value above
    // and are skipped here. Params init in declaration order (FOREACH_PARAM),
    // so a default that references an earlier param sees its loaded value.
    FOREACH_PARAM(Symbol & s, inst())
    {
        if (!s.everused() || s.typespec().is_structure())
            continue;
        if (s.connected())
            continue;
        if (s.interactive()
            && group().interactive_param_offset(layer, s.name()) >= 0)
            continue;
        const TypeSpec ts       = s.typespec();
        bool defaultval_initops = s.has_init_ops()
                                  && s.valuesource() == Symbol::DefaultVal;
        if (s.interpolated() && !ts.is_closure_based()) {
            // Bind interpolated (userdata) params from the renderer. Find this
            // symbol's userdata index, then call osl_bind_interpolated_param to
            // retrieve the value into the GroupData userdata slot and copy it
            // into the local. If no userdata is available (returns 0), fall back
            // to the param's default: its init ops if it has default-valued ones,
            // otherwise its constant default literal. Mirrors the interpolated
            // path in BackendLLVM::llvm_assign_initial_value.
            int ui = -1;
            for (int i = 0, e = (int)group().m_userdata_names.size(); i < e;
                 ++i)
                if (s.name() == group().m_userdata_names[i]
                    && equivalent(ts.simpletype(),
                                  group().m_userdata_types[i])) {
                    ui = i;
                    break;
                }
            OSL_DASSERT(ui >= 0);
            long long tdp  = OSL::bitcast<long long, TypeDesc>(ts.simpletype());
            std::string nm = s.cpp_safe_name();
            std::string sym = ts.is_array() ? fmtformat("(void*){}", nm)
                                            : fmtformat("(void*)&{}", nm);
            outputfmtln(
                "int {}__got = osl_bind_interpolated_param((void*)sg, "
                "OSL::ustring(\"{}\").hash(), {}LL, {}, (void*)gd->userdata{}_{}_, "
                "{}, {}, {}, (char*)&gd->userdata_init_flags[{}], {});",
                nm, s.name(), tdp, (int)group().m_userdata_derivs[ui], ui,
                group().m_userdata_names[ui], (int)s.has_derivs(), sym,
                s.derivsize(), ui, ui);
            outputfmtln("if (!{}__got) {{", nm);
            increment_indent();
            if (defaultval_initops)
                build_cpp_code(s.initbegin(), s.initend(), false);
            else
                outputfmtln("{} = {};", nm, cpp_const_literal_str(s));
            decrement_indent();
            outputfmtln("}}");
        } else if (defaultval_initops) {
            build_cpp_code(s.initbegin(), s.initend(), false);
        }
    }

    // Emit shader ops (main body only; params' init ops ran above).
    build_cpp_code(inst()->maincodebegin(), int(inst()->ops().size()), false);

    // Landing label for a shader-scope return/exit: it precedes the write-back
    // passes below so an early return still publishes the outputs computed so
    // far. Only emitted when a top-level return/exit exists (else the label
    // would be unused). `exit` ops always branch here; `return` ops outside any
    // inlined function do too (body_has_return ignores returns nested in
    // functioncalls).
    bool has_shader_exit = body_has_return(0, int(inst()->ops().size()));
    for (auto& o : inst()->ops())
        if (o.opname() == "exit")
            has_shader_exit = true;
    if (has_shader_exit)
        outputfmtln("cpp_layer_exit:;");

    // Write modified globals back to the ShaderGlobals struct. The JIT operates
    // through a pointer into sg, so its writes land automatically; here each
    // global was loaded into a local, so a written global must be copied back.
    // Most globals (u/v/P/...) are read-only inputs, but some shaders write
    // them — notably Ci, and P/N in a displacement shader.
    FOREACH_SYM(Symbol & s, inst())
    {
        if (s.symtype() != SymTypeGlobal || !s.everwritten())
            continue;
        std::string nm = s.cpp_safe_name();
        if (s.typespec().is_closure_based()) {
            // Closures are held as closure_color_t (const void*); the sg field
            // is ClosureColor*, so cast on store-back.
            outputfmtln("sg->{} = (OSL::ClosureColor*){};", s.name(), nm);
        } else if (sym_carries_derivs(s)) {
            // A Dual2-promoted global (e.g. P in a displacement shader) was
            // loaded as Dual2(sg->name, sg->dnamedx, sg->dnamedy); write its
            // value and derivatives back to the matching fields.
            std::string sn = s.name().string();
            outputfmtln(
                "sg->{} = {}.val(); sg->d{}dx = {}.dx(); sg->d{}dy = {}.dy();",
                sn, nm, sn, nm, sn, nm);
        } else {
            outputfmtln("sg->{} = {};", s.name(), nm);
        }
    }

    // Store non-local output params back to GroupData
    FOREACH_PARAM(Symbol & s, inst())
    {
        if (!s.everused())
            continue;
        if (s.symtype() == SymTypeOutputParam
            && !cpp_can_treat_param_as_local(s)) {
            TypeSpec ts = s.typespec();
            if (ts.is_structure())
                continue;
            if (s.typespec().is_array())
                outputfmtln("std::memcpy(gd->lay{}param_{}, {}, sizeof({}));",
                            layer, s.cpp_safe_name(), s.cpp_safe_name(),
                            s.cpp_safe_name());
            else
                outputfmtln("gd->lay{}param_{} = {};", layer, s.cpp_safe_name(),
                            s.cpp_safe_name());
        }
    }

    // Copy renderer-output params into the bound output buffer (output_base),
    // mirroring the JIT's "copy results to renderer outputs" pass. The host
    // reads shader outputs from this buffer at each symbol's symloc offset.
    FOREACH_PARAM(Symbol & s, inst())
    {
        if (!s.renderer_output())
            continue;
        const SymLocationDesc* symloc = group().find_symloc(s.name(),
                                                            inst()->layername(),
                                                            SymArena::Outputs);
        if (!symloc)
            continue;
        if (!equivalent(s.typespec(), symloc->type)
            || s.typespec().is_closure())
            continue;
        int size = int(symloc->type.size());
        if (symloc->derivs && s.has_derivs())
            size *= 3;  // also copy the derivs
        outputfmt(
            "{}std::memcpy((char*)output_base + ({}LL + {}LL * (long long)shadeindex),\n",
            indentstr(), (long long)symloc->offset, (long long)symloc->stride);
        outputfmt("{}            &{}, {});\n", indentstr(), s.cpp_safe_name(),
                  size);
        if (symloc->derivs && !s.has_derivs()) {
            // Output wants derivs but the source has none: zero the deriv area.
            int basesize = int(symloc->type.size());
            outputfmt(
                "{}std::memset((char*)output_base + ({}LL + {}LL * (long long)shadeindex) + {}, 0, {});\n",
                indentstr(), (long long)symloc->offset,
                (long long)symloc->stride, basesize, 2 * basesize);
        }
    }

    // Copy-down: propagate this layer's outputs to connected downstream inputs.
    // Mirrors the connection copy at the end of BackendLLVM::build_llvm_instance
    // (which calls llvm_assign_impl with the src/dst channels): a connection may
    // select a single source channel and/or write a single destination channel,
    // with a scalar source broadcast across an aggregate destination. A dest
    // aggregate whose channels are only partially connected is first set to its
    // default so the unconnected channels retain it.
    int nlayers = group().nlayers();
    for (int child_layer = layer + 1; child_layer < nlayers; ++child_layer) {
        ShaderInstance* child = group()[child_layer];
        if (child->unused())
            continue;
        int Nc = child->nconnections();
        // Destination symbols already default-initialized for partial coverage.
        std::vector<const Symbol*> inited;
        for (int c = 0; c < Nc; ++c) {
            const Connection& con = child->connection(c);
            if (con.srclayer != layer)
                continue;
            Symbol* srcsym = inst()->symbol(con.src.param);
            Symbol* dstsym = child->symbol(con.dst.param);
            TypeSpec dts   = dstsym->typespec();
            if (dts.is_structure())
                continue;
            std::string dstbase = fmtformat("gd->lay{}param_{}", child_layer,
                                            dstsym->cpp_safe_name());
            // Closures and whole arrays copy as a unit (the optimizer never
            // produces array-element connections, so these are always complete).
            // Arrays — including closure arrays (arrays of pointers) — memcpy;
            // a scalar closure copies its pointer. Check is_array() first so a
            // closure array doesn't take the non-assignable scalar path.
            if (dts.is_array()) {
                outputfmtln("std::memcpy({}, {}, sizeof({}));", dstbase,
                            srcsym->cpp_safe_name(), srcsym->cpp_safe_name());
                continue;
            }
            if (dts.is_closure_based()) {
                outputfmtln("{} = {};", dstbase, srcsym->cpp_safe_name());
                continue;
            }

            const int srcchan = con.src.channel;
            const int dstchan = con.dst.channel;
            TypeDesc dt       = dts.simpletype();
            const int agg     = dt.aggregate;

            // Partial-init: a destination aggregate whose channels are not all
            // connected gets its default first, so unconnected channels keep it
            // (mirrors the JIT's initedsyms/ninit logic).
            if (dstchan != -1
                && std::find(inited.begin(), inited.end(), dstsym)
                       == inited.end()) {
                inited.push_back(dstsym);
                uint32_t covered = 0;
                bool whole       = false;
                for (int rc = 0; rc < Nc; ++rc) {
                    const Connection& nx = child->connection(rc);
                    if (child->symbol(nx.dst.param) != dstsym)
                        continue;
                    if (nx.dst.channel == -1) {
                        whole = true;
                        break;
                    }
                    covered |= (1u << nx.dst.channel);
                }
                if (!whole && OSL::popcount(covered) < agg)
                    outputfmtln("{} = {};", dstbase,
                                cpp_const_literal_str(*dstsym));
            }

            // A Dual2 destination/source's component math targets its value.
            std::string dstagg = sym_carries_derivs(*dstsym)
                                     ? dstbase + ".val()"
                                     : dstbase;
            std::string sv     = srcsym->cpp_safe_name();
            if (sym_carries_derivs(*srcsym))
                sv += ".val()";
            TypeDesc st        = srcsym->typespec().simpletype();
            const bool src_agg = st.aggregate > 1;
            auto dcomp         = [&](int i) {
                return dt.aggregate == TypeDesc::MATRIX44
                                   ? fmtformat("{}[{}][{}]", dstagg, i / 4, i % 4)
                                   : fmtformat("{}[{}]", dstagg, i);
            };
            auto sval = [&](int comp) -> std::string {
                if (!src_agg)
                    return sv;
                return st.aggregate == TypeDesc::MATRIX44
                           ? fmtformat("{}[{}][{}]", sv, comp / 4, comp % 4)
                           : fmtformat("{}[{}]", sv, comp);
            };

            const bool singlechan = (srcchan != -1) || (dstchan != -1);
            if (!singlechan) {
                // Whole -> whole. Matching aggregate (or scalar) copies as a
                // unit; a scalar source into an aggregate dest broadcasts.
                if (agg == 1 || src_agg)
                    outputfmtln("{} = {};", dstbase, sv);
                else
                    for (int i = 0; i < agg; ++i)
                        outputfmtln("{} = {};", dcomp(i), sv);
            } else {
                // Connect a single source channel (a float if src is scalar).
                std::string v = sval(srcchan == -1 ? 0 : srcchan);
                if (dstchan != -1)
                    outputfmtln("{} = {};", dcomp(dstchan), v);
                else if (agg == 1)
                    outputfmtln("{} = {};", dstbase, v);
                else
                    for (int i = 0; i < agg; ++i)
                        outputfmtln("{} = {};", dcomp(i), v);
            }
        }
    }

    decrement_indent();
    outputfmt("{}}}\n\n", indentstr());
}



void
BackendCpp::generate_group_entry()
{
    int nlayers  = group().nlayers();
    int num_used = 0;
    for (int i = 0; i < nlayers; ++i)
        if (m_layer_remap[i] >= 0)
            ++num_used;

    std::string group_name = group().name().string();
    std::string ptr        = lang_ptr_syntax();
    std::string lp         = lang_linkage_prefix();
    if (!lp.empty())
        lp += " ";

    // ABI version export
    outputfmt(
        "{}int osl_cpp_abi_version() {{ return OSL::OSL_CPP_ABI_VERSION; }}\n\n",
        lp);

    // Group entry: signature matches RunLLVMGroupFunc (all void*)
    outputfmt("{}void osl_init_group_{}(\n", lp, group_name);
    outputfmt("    void{0} shaderglobals_ptr, void{0} heap_arena_ptr,\n", ptr);
    outputfmt(
        "    void{0} userdata_base_pointer, void{0} output_base_pointer,\n",
        ptr);
    outputfmt("    int shadeindex, void{0} interactive_params_ptr)\n", ptr);
    outputfmtln("{{");
    increment_indent();

    outputfmtln(
        "OSL::ShaderGlobals{} sg = (OSL::ShaderGlobals{})shaderglobals_ptr;",
        ptr, ptr);
    outputfmtln("GroupData{} gd = (GroupData{})heap_arena_ptr;", ptr, ptr);

    // Zero all layer runflags
    outputfmtln("for (int i = 0; i < {}; ++i) gd->layer_runflags[i] = false;",
                num_used);

    // Zero the userdata "initialized" flags so the first interpolated-param bind
    // in each layer triggers a get_userdata retrieval (status 0 = not yet
    // retrieved). Mirrors the JIT group-init memset of userdata_initialized.
    int nuserdata = (int)group().m_userdata_names.size();
    if (nuserdata)
        outputfmtln(
            "for (int i = 0; i < {}; ++i) gd->userdata_init_flags[i] = 0;",
            nuserdata);

    // Dispatch the non-lazy layers in dependency order. A run_lazily() layer is
    // NOT run here — it executes on demand when a downstream layer reads one of
    // its connected outputs (see cpp_gen_useparam, which mirrors
    // BackendLLVM::llvm_run_connected_layers). Each layer sets its own run-flag
    // at entry, so the guard here only avoids a redundant call.
    for (int layer = 0; layer < nlayers; ++layer) {
        ShaderInstance* linst = group()[layer];
        if (linst->unused() || linst->empty_instance())
            continue;
        if (linst->run_lazily())
            continue;
        int ri = m_layer_remap[layer];
        if (ri < 0)
            continue;
        std::string layer_func = fmtformat("osl_layer_group_{}_name_{}",
                                           group_name, linst->layername());
        outputfmtln("if (!gd->layer_runflags[{}]) {{", ri);
        increment_indent();
        outputfmt("{}{}(sg, gd, userdata_base_pointer, output_base_pointer,\n",
                  indentstr(), layer_func);
        outputfmt("{}    shadeindex, interactive_params_ptr);\n", indentstr());
        decrement_indent();
        outputfmtln("}}");
    }

    decrement_indent();
    outputfmt("}}\n\n");
}



void
BackendCpp::run()
{
    outputfmt("{}\n\n", lang_preamble());

    // Compact run-flag index over used layers only (shared by layer funcs, the
    // group entry, and cpp_gen_useparam's lazy upstream dispatch).
    int nlayers = (int)group().nlayers();
    m_layer_remap.assign(nlayers, -1);
    int num_used = 0;
    for (int i = 0; i < nlayers; ++i)
        if (!group()[i]->unused())
            m_layer_remap[i] = num_used++;

    generate_groupdata_struct();

    for (int layer = 0; layer < nlayers; ++layer)
        generate_layer_func(layer);

    generate_group_entry();
}



// debug_nan: after an op that writes a float-based value, emit an
// osl_naninf_check on each written float argument (mirrors the JIT's
// llvm_generate_debugnan). Partial-write ops (aassign/compassign/mxcompassign)
// restrict the check to the element actually written to avoid false positives on
// untouched elements.
static void
cpp_generate_debugnan(BackendCpp& rop, int opnum)
{
    const Opcode& op(rop.inst()->ops()[opnum]);
    for (int i = 0; i < op.nargs(); ++i) {
        if (!op.argwrite(i))
            continue;
        Symbol& sym(*rop.opargsym(op, i));
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT)
            continue;
        int ncomps         = int(t.numelements() * t.aggregate);
        std::string offset = "0";
        std::string ncheck = fmtformat("{}", ncomps);
        if (op.opname() == "aassign") {
            std::string ind = rop.cpp_value_str(*rop.opargsym(op, 1));
            int agg         = t.aggregate;
            offset          = agg == 1 ? ind : fmtformat("({} * {})", ind, agg);
            ncheck          = fmtformat("{}", agg);
        } else if (op.opname() == "compassign") {
            offset = rop.cpp_value_str(*rop.opargsym(op, 1));
            ncheck = "1";
        } else if (op.opname() == "mxcompassign") {
            std::string row = rop.cpp_value_str(*rop.opargsym(op, 1));
            std::string col = rop.cpp_value_str(*rop.opargsym(op, 2));
            offset          = fmtformat("({} * 4 + {})", row, col);
            ncheck          = "1";
        }
        rop.outputfmtln(
            "osl_naninf_check({}, (void*)&{}, {}, (void*)sg, {}, {}, {}, {}, {}, {});",
            ncomps, sym.cpp_safe_name(), (int)sym.has_derivs(),
            rop.cpp_string_literal_rep(op.sourcefile()), op.sourceline(),
            rop.cpp_string_literal_rep(sym.unmangled()), offset, ncheck,
            rop.cpp_string_literal_rep(op.opname()));
    }
}



// debug_uninit: initialize a local/temp to the "uninitialized" marker (NaN for
// float-based, INT_MIN for int, the uninitialized string for strings) so a
// subsequent read of an unwritten value can be detected. Mirrors the debug_uninit
// branch of BackendLLVM::llvm_assign_initial_value. Only the value components are
// marked (derivs are not checked). Emitted only when debug_uninit is enabled.
void
BackendCpp::cpp_uninit_marker_init(const Symbol& s)
{
    const TypeDesc t = s.typespec().simpletype();
    if (s.typespec().is_closure_based())
        return;
    std::string nm = s.cpp_safe_name();
    int n          = std::max(1, int(t.numelements()));   // array elements
    int ncomp      = int(t.numelements() * t.aggregate);  // total scalar slots
    if (t.basetype == TypeDesc::FLOAT) {
        static std::string nan("std::numeric_limits<float>::quiet_NaN()");
        if (sym_carries_derivs(s)) {
            // Mark the value of each element (a Dual2<elem>); derivs unchecked.
            std::string elem  = lang_type_name(t);  // e.g. "float"
            std::string vctor = t.aggregate == 1
                                    ? nan
                                    : fmtformat("{}({}, {}, {})", elem, nan,
                                                nan, nan);
            if (t.arraylen > 0)
                outputfmtln(
                    "for (int ___k=0;___k<{};++___k) {}[___k].val() = {};", n,
                    nm, vctor);
            else
                outputfmtln("{}.val() = {};", nm, vctor);
        } else {
            // Contiguous value storage: set every float slot to NaN.
            outputfmtln(
                "{{ float* ___u = (float*)&{}; for (int ___k=0;___k<{};++___k) ___u[___k] = {}; }}",
                nm, ncomp, nan);
        }
    } else if (t.basetype == TypeDesc::INT) {
        outputfmtln(
            "{{ int* ___u = (int*)&{}; for (int ___k=0;___k<{};++___k) ___u[___k] = std::numeric_limits<int>::min(); }}",
            nm, ncomp);
    } else if (t.basetype == TypeDesc::STRING
               || t.basetype == TypeDesc::USTRINGHASH) {
        std::string mk
            = "OSL::ustringhash::from_hash(OSL::ustring(\"!!!uninitialized!!!\").hash())";
        if (t.arraylen > 0)
            outputfmtln("for (int ___k=0;___k<{};++___k) {}[___k] = {};", n, nm,
                        mk);
        else
            outputfmtln("{} = {};", nm, mk);
    }
}



// debug_uninit: before an op reads its arguments, check each read value for the
// uninitialized marker via osl_uninit_check, reporting an uninitialized read.
// Mirrors BackendLLVM::llvm_generate_debug_uninit, including the partial-read
// special cases (aref/compref/mxcompref read one element; spline limits the knot
// check to the knot count).
void
BackendCpp::cpp_generate_debug_uninit(int opnum)
{
    const Opcode& op(inst()->ops()[opnum]);
    // useparam's args are by definition not yet set before the op runs.
    if (op.opname() == "useparam")
        return;
    for (int i = 0; i < op.nargs(); ++i) {
        if (!op.argread(i))
            continue;
        Symbol& sym(*opargsym(op, i));
        // Constants are always initialized (and scalar consts are inlined with no
        // address); only locals/temps ever hold the uninitialized marker.
        if (sym.symtype() == SymTypeConst)
            continue;
        if (sym.typespec().is_closure_based())
            continue;
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT && t.basetype != TypeDesc::INT
            && t.basetype != TypeDesc::STRING
            && t.basetype != TypeDesc::USTRINGHASH)
            continue;
        // The loop/condition temp may not have had its initializer run yet.
        if (op.opname() == "for" && i == 0)
            continue;
        if ((op.opname() == "dowhile" || op.opname() == "while") && i == 0
            && op.jump(0) != op.jump(1))
            continue;

        std::string offset = "0";
        std::string ncheck = fmtformat("{}",
                                       int(t.numelements() * t.aggregate));
        if (op.opname() == "aref" && i == 1) {
            std::string ind = cpp_value_str(*opargsym(op, 2));
            int agg         = t.aggregate;
            offset          = agg == 1 ? ind : fmtformat("({} * {})", ind, agg);
            ncheck          = fmtformat("{}", agg);
        } else if (op.opname() == "compref" && i == 1) {
            offset = cpp_value_str(*opargsym(op, 2));
            ncheck = "1";
        } else if (op.opname() == "mxcompref" && i == 1) {
            std::string row = cpp_value_str(*opargsym(op, 2));
            std::string col = cpp_value_str(*opargsym(op, 3));
            offset          = fmtformat("({} * 4 + {})", row, col);
            ncheck          = "1";
        } else if ((op.opname() == "spline" || op.opname() == "splineinverse")
                   && i == 4 && op.nargs() == 5) {
            ncheck = cpp_value_str(*opargsym(op, 3));
        }
        long long tdp  = OSL::bitcast<long long, TypeDesc>(t);
        std::string vp = sym.typespec().is_array()
                             ? fmtformat("(void*){}", sym.cpp_safe_name())
                             : fmtformat("(void*)&{}", sym.cpp_safe_name());
        outputfmtln(
            "osl_uninit_check({}LL, {}, (void*)sg, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});",
            tdp, vp, cpp_string_literal_rep(op.sourcefile()), op.sourceline(),
            cpp_string_literal_rep(group().name()), layer(),
            cpp_string_literal_rep(inst()->layername()),
            cpp_string_literal_rep(inst()->shadername()), opnum,
            cpp_string_literal_rep(op.opname()), i,
            cpp_string_literal_rep(sym.unmangled()), offset, ncheck);
    }
}



void
BackendCpp::build_cpp_code(int opbegin, int opend, bool do_indent_block)
{
    if (do_indent_block) {
        outputfmtln("{{");
        increment_indent();
    }
    for (int opnum = opbegin; opnum < opend; ++opnum) {
        const Opcode& op(inst()->ops()[opnum]);
        if (opnum == inst()->maincodebegin())
            outputfmtln("// (main)");
        // With debug_uninit enabled, check this op's read args for the
        // uninitialized marker before the op runs.
        if (shadingsys().debug_uninit())
            cpp_generate_debug_uninit(opnum);

        auto* opdesc = shadingsys().op_descriptor(op.opname());
        if (opdesc && opdesc->cppgen) {
            // If the opcode has a C++ generator, call it
            if (!opdesc->cppgen(*this, opnum))
                outputfmtln("// Cpp {} FAILED", op.opname());
        } else {
            // Otherwise, generate the default C++ code for it
            outputfmtln("// NO CPP GENERATOR FOR {}", op.opname());
        }

        // With debug_nan enabled, check this op's float writes for NaN/Inf.
        if (shadingsys().debug_nan())
            cpp_generate_debugnan(*this, opnum);

        // If the op we coded jumps around, skip past its recursive block
        // executions.
        int next = op.farthest_jump();
        if (next >= 0)
            opnum = next - 1;
    }
    if (do_indent_block) {
        decrement_indent();
        outputfmtln("}}");
    }
}



// C++ code generator for no-ops: things that should be silent like giraffes
// when generating C++ code.
bool
cpp_gen_nop(BackendCpp& rop, int opnum)
{
    return true;
}



// C++ code generator for functioncall / functioncall_nr.
//
// In OSL's IR, an inlined function call is represented as:
//   functioncall "name"  [jump(0) = first op after the body]
//     <body ops>
// The LLVM backend uses build_llvm_code(opnum+1, jump(0)) to emit the body.
// We do the same for C++: emit the body inline, then return.  build_cpp_code's
// farthest_jump mechanism will then advance opnum past the body automatically.
bool
cpp_gen_functioncall(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int body_begin = opnum + 1;
    int body_end   = op.jump(0);

    // A 'return' inside the body must jump to the end of THIS inlined body, not
    // out of the whole layer function. If the body contains a return, emit a
    // goto-label after it and push it as the function's return target.
    std::string ret_lbl;
    if (rop.body_has_return(body_begin, body_end))
        ret_lbl = fmtformat("cpp_func_return_{}", rop.new_loop_label_id());
    rop.push_func_context(ret_lbl);

    rop.build_cpp_code(body_begin, body_end, false);

    rop.pop_func_context();
    if (!ret_lbl.empty())
        rop.outputfmtln("{}:;", ret_lbl);
    return true;
}



// C++ code generator for "generic" functions.
//
// Builds the same type-mangled name as llvm_gen_generic:
//   osl_<opname>_<per-arg suffix>
// where suffix chars are: f=float, v=triple, m=matrix, s=string, i=int.
// The C++ backend does not track derivatives, so no 'd' prefix is emitted.
//
// Calling conventions (matching the osl_* ABI in liboslexec):
//   Scalar result  -> function returns value:  R = osl_cos_ff(a);
//   Aggregate result -> function returns void, result is first void* arg:
//                       osl_cos_vv((void*)&R, (void*)&a);
// Within an aggregate-result call, scalar args are passed by value and
// aggregate args are passed as void*.
bool
cpp_gen_generic(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 1);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    int nargs = op.nargs();

    // Derivative bookkeeping, mirroring llvm_gen_generic: an input arg with
    // derivatives (matrices excepted) triggers the deriv-aware osl_* variant,
    // suppressed for the ops whose derivatives are always zero.
    bool any_deriv_args = false;
    for (int i = 1; i < nargs; ++i) {
        const Symbol* s = rop.opargsym(op, i);
        any_deriv_args |= (s->has_derivs() && !s->typespec().is_matrix());
    }
    ustring opn = op.opname();
    if (any_deriv_args
        && (opn == "logb" || opn == "floor" || opn == "ceil" || opn == "round"
            || opn == "step" || opn == "trunc" || opn == "sign"))
        any_deriv_args = false;
    bool derivs_call = R.has_derivs() && any_deriv_args;

    // Per-arg: does the called variant take this arg with derivatives?
    auto arg_derivs = [&](const Symbol& s) {
        return derivs_call && s.has_derivs();
    };

    // Build mangled name: osl_<opname>_<typecode per arg including result>.
    std::string name = std::string("osl_") + opn.string() + "_";
    for (int i = 0; i < nargs; ++i) {
        const Symbol* s = rop.opargsym(op, i);
        name += s->arg_typecode(arg_derivs(*s));
    }

    // Argument expression: pass a void* to the (Dual2) storage for aggregates
    // and for any arg the variant takes with derivatives; otherwise by value
    // (stripping a Dual2 scalar to .val()).
    auto arg_str = [&](const Symbol& s) -> std::string {
        if (s.typespec().aggregate() != TypeDesc::SCALAR || arg_derivs(s))
            return std::string("(void*)&") + s.cpp_safe_name();
        // String scalars are ustringhash variables but osl_* take them by value
        // as ustringhash_pod, so pass the hash (const string consts are already
        // the raw uint64 pod).
        if (s.typespec().is_string())
            return rop.cpp_spacename_pod(s);
        std::string str = rop.cpp_value_str(s);
        if (rop.sym_carries_derivs(s))
            str += ".val()";
        return str;
    };

    bool scalar_result = (R.typespec().aggregate() == TypeDesc::SCALAR);

    if (derivs_call) {
        // Deriv-aware variant: every arg (result first) is passed positionally,
        // Dual2/aggregate storage by void*.  The function writes all of R.
        rop.outputfmt("{}{}(", rop.indentstr(), name);
        for (int i = 0; i < nargs; ++i)
            rop.outputfmt("{}{}", i ? ", " : "", arg_str(*rop.opargsym(op, i)));
        rop.outputfmt(");\n");
    } else if (scalar_result) {
        // osl_name_ff(a, b, ...) — result NOT in arg list, returned by value.
        // A Dual2<float> result is constructed from the float (derivs zeroed).
        // A string result comes back as ustringhash_pod (uint64); wrap it so it
        // assigns to the ustringhash result variable.
        bool str_result = R.typespec().is_string();
        rop.outputfmt("{}{} = {}{}(", rop.indentstr(), R.cpp_safe_name(),
                      str_result ? "OSL::ustringhash::from_hash(" : "", name);
        for (int a = 1; a < nargs; ++a)
            rop.outputfmt("{}{}", a > 1 ? ", " : "",
                          arg_str(*rop.opargsym(op, a)));
        rop.outputfmt("){};\n", str_result ? ")" : "");
    } else {
        // osl_name_vv((void*)&R, ...) — result IS first arg as void*.
        rop.outputfmt("{}{}((void*)&{}", rop.indentstr(), name,
                      R.cpp_safe_name());
        for (int a = 1; a < nargs; ++a)
            rop.outputfmt(", {}", arg_str(*rop.opargsym(op, a)));
        rop.outputfmt(");\n");
        // The non-deriv variant wrote only R's value; a deriv-carrying triple
        // result must have its (now-garbage) partials zeroed, matching the JIT's
        // llvm_zero_derivs.
        if (rop.sym_carries_derivs(R))
            rop.outputfmtln("{} = {}({}.val());", R.cpp_safe_name(),
                            rop.lang_sym_type_name(R), R.cpp_safe_name());
    }
    return true;
}



// Dedicated generator for noise()/snoise()/pnoise()/psnoise(), mirroring
// llvm_gen_noise.  A constant noise-type name ("cell", "perlin", …) is
// canonicalized into the osl_* function symbol rather than passed as an
// argument.  Two cases also take a NoiseParams options struct, the ShaderGlobals
// pointer, and the noise-type name as a leading ustringhash_pod argument:
//   * gabor noise (constant name "gabor"), and
//   * generic noise — the name is not a compile-time constant.
bool
cpp_gen_noise(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    ustring opname = op.opname();
    bool periodic  = (opname == "pnoise" || opname == "psnoise");

    int arg        = 0;
    Symbol& Result = *rop.opargsym(op, arg++);
    int outdim     = Result.typespec().is_triple() ? 3 : 1;
    Symbol* Name   = rop.opargsym(op, arg++);
    ustring name;
    if (Name->typespec().is_string()) {
        name = Name->is_constant() ? Name->get_string() : ustring();
    } else {
        // Old-style unnamed noise/pnoise: the op name is the noise type.
        --arg;
        Name = nullptr;
        name = opname;
    }

    Symbol* S    = rop.opargsym(op, arg++);
    Symbol* T    = nullptr;
    Symbol* Sper = nullptr;
    Symbol* Tper = nullptr;
    int indim    = S->typespec().is_triple() ? 3 : 1;
    bool derivs  = S->has_derivs();

    if (periodic) {
        if (op.nargs() > (arg + 1)
            && (rop.opargsym(op, arg + 1)->typespec().is_float()
                || rop.opargsym(op, arg + 1)->typespec().is_triple())) {
            ++indim;
            T = rop.opargsym(op, arg++);
            derivs |= T->has_derivs();
        }
        Sper = rop.opargsym(op, arg++);
        if (indim == 2 || indim == 4)
            Tper = rop.opargsym(op, arg++);
    } else {
        if (op.nargs() > arg && rop.opargsym(op, arg)->typespec().is_float()) {
            ++indim;
            T = rop.opargsym(op, arg++);
            derivs |= T->has_derivs();
        }
    }
    derivs &= Result.has_derivs();  // ignore derivs if result doesn't need them
    int first_optional_arg = arg;   // remaining args are (token, value) pairs

    // Canonicalize the noise-type name into the osl_* function base name and
    // decide whether this is an options-taking (generic/gabor) call.
    bool pass_name = false, pass_sg = false, pass_options = false;
    if (name.empty()) {
        // Name not a compile-time constant: generic noise dispatch.
        name = periodic ? ustring("genericpnoise") : ustring("genericnoise");
        pass_name    = true;
        pass_sg      = true;
        pass_options = true;
        derivs       = true;  // always take derivs when the type is unknown
    } else if (name == "perlin" || name == "snoise" || name == "psnoise") {
        name = periodic ? ustring("psnoise") : ustring("snoise");
    } else if (name == "uperlin" || name == "noise" || name == "pnoise") {
        name = periodic ? ustring("pnoise") : ustring("noise");
    } else if (name == "cell" || name == "cellnoise") {
        name   = periodic ? ustring("pcellnoise") : ustring("cellnoise");
        derivs = false;  // cell noise derivs are always zero
    } else if (name == "hash" || name == "hashnoise") {
        name   = periodic ? ustring("phashnoise") : ustring("hashnoise");
        derivs = false;  // hash noise derivs are always zero
    } else if (name == "simplex" && !periodic) {
        name = ustring("simplexnoise");
    } else if (name == "usimplex" && !periodic) {
        name = ustring("usimplexnoise");
    } else if (name == "gabor") {
        name      = periodic ? ustring("gaborpnoise") : ustring("gabornoise");
        pass_name = true;
        pass_sg   = true;
        pass_options = true;
        derivs       = true;
    } else {
        rop.shadingcontext()->errorfmt(
            "{}noise type \"{}\" is unknown, called from ({}:{})",
            periodic ? "periodic " : "", name, op.sourcefile(),
            op.sourceline());
        return false;
    }

    // Build a NoiseParams options struct from the trailing (token, value) pairs.
    std::string optvar;
    if (pass_options) {
        optvar = fmtformat("_noiseopt{}", opnum);
        rop.outputfmtln("OSL::pvt::NoiseParams {};", optvar);
        rop.outputfmtln("osl_init_noise_options((void*)sg, (void*)&{});",
                        optvar);
        for (int a = first_optional_arg; a + 1 < op.nargs(); a += 2) {
            Symbol& Tok = *rop.opargsym(op, a);
            Symbol& Val = *rop.opargsym(op, a + 1);
            if (!Tok.is_constant() || !Tok.typespec().is_string())
                continue;
            ustring tok = Tok.get_string();
            if (tok.empty())
                continue;
            if (tok == "anisotropic" && Val.typespec().is_int())
                rop.outputfmtln(
                    "osl_noiseparams_set_anisotropic((void*)&{}, {});", optvar,
                    rop.cpp_value_str(Val));
            else if (tok == "do_filter" && Val.typespec().is_int())
                rop.outputfmtln("osl_noiseparams_set_do_filter((void*)&{}, {});",
                                optvar, rop.cpp_value_str(Val));
            else if (tok == "direction" && Val.typespec().is_triple())
                rop.outputfmtln(
                    "osl_noiseparams_set_direction((void*)&{}, (void*)&{});",
                    optvar, Val.cpp_safe_name());
            else if (tok == "bandwidth"
                     && (Val.typespec().is_float() || Val.typespec().is_int()))
                rop.outputfmtln("osl_noiseparams_set_bandwidth((void*)&{}, {});",
                                optvar, rop.cpp_value_str(Val));
            else if (tok == "impulses"
                     && (Val.typespec().is_float() || Val.typespec().is_int()))
                rop.outputfmtln("osl_noiseparams_set_impulses((void*)&{}, {});",
                                optvar, rop.cpp_value_str(Val));
        }
    }

    // Build the function name (mirrors llvm_gen_noise typecode assembly).
    std::string funcname = "osl_" + name.string() + "_"
                           + Result.arg_typecode(derivs);
    funcname += S->arg_typecode(derivs);
    if (T)
        funcname += T->arg_typecode(derivs);
    if (periodic) {
        funcname += Sper->arg_typecode(false);
        if (Tper)
            funcname += Tper->arg_typecode(false);
    }

    // An argument: a void* to the (Dual2) storage for aggregates and for any
    // arg the variant takes with derivatives; otherwise by value.
    auto arg_str = [&](const Symbol& s, bool with_derivs) -> std::string {
        if (s.typespec().aggregate() != TypeDesc::SCALAR || with_derivs)
            return std::string("(void*)&") + s.cpp_safe_name();
        std::string str = rop.cpp_value_str(s);
        if (rop.sym_carries_derivs(s))
            str += ".val()";
        return str;
    };

    // Calling convention: a triple result, or any result-with-derivs call, is
    // written through a leading result pointer (function returns void); a plain
    // float result is returned by value.  When the variant produces derivs but
    // our Result has none, write into a Dual2 temp and copy the value back.
    bool result_ptr = (outdim == 3 || derivs);
    bool need_temp  = derivs && !Result.has_derivs();
    std::string resvar;
    if (need_temp) {
        resvar = fmtformat("_noiseres{}", opnum);
        rop.outputfmtln("{} {};",
                        outdim == 3 ? "OSL::Dual2<OSL::Vec3>"
                                    : "OSL::Dual2<float>",
                        resvar);
    } else {
        resvar = Result.cpp_safe_name();
    }

    // Assemble the positional argument list.
    std::vector<std::string> args;
    if (pass_name) {
        // The noise-type name is passed as a ustringhash_pod (the raw hash).
        args.push_back(Name->is_constant() ? rop.cpp_value_str(*Name)
                                           : Name->cpp_safe_name() + ".hash()");
    }
    if (result_ptr)
        args.push_back(std::string("(void*)&") + resvar);
    args.push_back(arg_str(*S, derivs));
    if (T)
        args.push_back(arg_str(*T, derivs));
    if (periodic) {
        args.push_back(arg_str(*Sper, false));
        if (Tper)
            args.push_back(arg_str(*Tper, false));
    }
    if (pass_sg)
        args.push_back("(void*)sg");
    if (pass_options)
        args.push_back(std::string("(void*)&") + optvar);

    std::string arglist;
    for (size_t i = 0; i < args.size(); ++i)
        arglist += (i ? ", " : "") + args[i];

    if (result_ptr)
        rop.outputfmtln("{}({});", funcname, arglist);
    else
        rop.outputfmtln("{} = {}({});", resvar, funcname, arglist);

    if (need_temp)
        rop.outputfmtln("{} = {}.val();", Result.cpp_safe_name(), resvar);

    // Result carries derivs but we called a value-only variant: zero the
    // partials (mirrors llvm_zero_derivs / the generic generator).
    if (Result.has_derivs() && !derivs && rop.sym_carries_derivs(Result))
        rop.outputfmtln("{} = {}({}.val());", Result.cpp_safe_name(),
                        rop.lang_sym_type_name(Result), Result.cpp_safe_name());
    return true;
}



// C++ code generator for "generic" functions: just express it as a function
// call like:    result = osl_func(arg1, ...);
bool
cpp_gen_if(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym(op, 0);

    // Then block
    rop.outputfmtln("if ({})", rop.cpp_value_str(cond));
    rop.build_cpp_code(opnum + 1, op.jump(0));
    if (op.jump(0) != op.jump(1)) {
        rop.outputfmtln("else");
        rop.build_cpp_code(op.jump(0), op.jump(1));
    }
    return true;
}



// Whole-array copy R = A. When R and A have the same deriv-ness their storage
// layout matches (both Dual2<elem>[N] or both elem[N]), so a memcpy is correct.
// When deriv-ness differs the layouts differ (Dual2<elem> is 3x an elem and
// interleaved), so copy element-wise: a value source promotes to Dual2 (derivs
// zeroed) via Dual2's implicit ctor; a Dual2 source assigned to a value array
// takes .val().
void
BackendCpp::cpp_array_copy(const Symbol& R, const Symbol& A)
{
    // Mismatched array lengths copy only min(dst,src) elements, leaving the
    // destination's trailing elements unchanged (mirrors the std::min in
    // BackendLLVM::llvm_assign_impl). Using sizeof(R) would over-read a shorter
    // source and clobber the retained trailing elements with garbage.
    const size_t relems = R.typespec().simpletype().numelements();
    const size_t aelems = A.typespec().simpletype().numelements();
    const size_t n      = std::max(size_t(1), std::min(relems, aelems));
    if (sym_carries_derivs(R) == sym_carries_derivs(A)) {
        // Same deriv-ness => identical element layout; one memcpy of n elements.
        // sizeof(R[0]) accounts for the per-element size including Dual2 derivs.
        outputfmtln("std::memcpy({}, {}, {} * sizeof({}[0]));",
                    R.cpp_safe_name(), cpp_value_str(A), n, R.cpp_safe_name());
        return;
    }
    std::string rhs = (sym_carries_derivs(A) && !sym_carries_derivs(R))
                          ? fmtformat("{}[___i].val()", cpp_value_str(A))
                          : fmtformat("{}[___i]", cpp_value_str(A));
    outputfmtln("for (int ___i = 0; ___i < {}; ++___i) {}[___i] = {};", n,
                R.cpp_safe_name(), rhs);
}



bool
cpp_gen_assign(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    // Helper: get the value string for A, stripping a Dual2 to .val() when R is
    // not itself a Dual2 (otherwise the Dual2 -> plain assignment won't compile).
    auto a_val = [&]() -> std::string {
        std::string s = rop.cpp_value_str(A);
        if (rop.sym_carries_derivs(A) && !rop.sym_carries_derivs(R))
            s += ".val()";
        return s;
    };
    // Assignment to an array destination.
    if (R.typespec().is_array()) {
        if (A.typespec().is_array()) {
            // array = array: copy the storage (C arrays are not assignable).
            rop.cpp_array_copy(R, A);
        } else {
            // array = scalar/aggregate: OSL's `assign` initializes only
            // min(Rlen, srclen) = 1 element here, i.e. element 0 (an index-0
            // store with the index elided). Mirror that with an element-0 store
            // using the same scalar/aggregate conversion as the non-array path.
            std::string idx0 = R.cpp_safe_name() + "[0]";
            TypeSpec et      = R.typespec().elementtype();
            if (et.is_triple() && A.typespec().aggregate() == TypeDesc::SCALAR
                && !rop.sym_carries_derivs(R)) {
                // Broadcast a scalar across the triple element's components.
                std::string v = a_val();
                rop.outputfmtln("{} = {}({}, {}, {});", idx0,
                                rop.lang_type_name(et.simpletype()), v, v, v);
            } else if (et.is_string() && A.symtype() == SymTypeConst) {
                rop.outputfmtln("{} = OSL::ustringhash::from_hash({});", idx0,
                                rop.cpp_value_str(A));
            } else {
                rop.outputfmtln("{} = {};", idx0, a_val());
            }
        }
        return true;
    }
    // Closure assign: a closure is a pointer.  Copy from another closure, or set
    // null from a numeric-zero constant (`closure color c = 0`) — `= 0.0f` would
    // not convert to a pointer.
    if (R.typespec().is_closure_based()) {
        if (A.typespec().is_closure_based())
            rop.outputfmtln("{} = {};", R.cpp_safe_name(),
                            rop.cpp_value_str(A));
        else
            rop.outputfmtln("{} = nullptr;", R.cpp_safe_name());
        return true;
    }
    // Matrix = scalar: set the diagonal to the scalar (m=f / m=i), off-diagonal
    // to zero — Imath's Matrix44(T) would set every element instead.
    if (R.typespec().is_matrix()
        && A.typespec().aggregate() == TypeDesc::SCALAR) {
        rop.cpp_emit_matrix_diagonal(R, a_val());
        return true;
    }
    // Triple = scalar: Color3/Vec3 have no implicit ctor from scalar, so
    // broadcast via the 3-arg constructor (deriv-aware when either side carries
    // derivatives).
    if (R.typespec().is_triple()
        && A.typespec().aggregate() == TypeDesc::SCALAR) {
        rop.outputfmtln("{} = {};", R.cpp_safe_name(),
                        rop.cpp_triple_ctor(R, &A, &A, &A));
    } else if (R.typespec().is_string() && A.symtype() == SymTypeConst) {
        // String variables are ustringhash; a string constant is a raw uint64_t
        // hash. Construct the ustringhash from the hash to assign it.
        rop.outputfmtln("{} = OSL::ustringhash::from_hash({});",
                        R.cpp_safe_name(), rop.cpp_value_str(A));
    } else {
        rop.outputfmtln("{} = {};", R.cpp_safe_name(), a_val());
    }
    return true;
}



bool
cpp_gen_construct(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 2);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    rop.outputfmt("{}{} = {}(", rop.indentstr(), R.cpp_safe_name(),
                  rop.lang_sym_type_name(R));
    int nargs = op.nargs() - 1;
    for (int a = 0; a < nargs; ++a) {
        Symbol& A(*rop.inst()->argsymbol(op.firstarg() + a + 1));
        std::string av = rop.cpp_value_str(A);
        // R is Dual2<float> only when scalar float with derivs; otherwise
        // any Dual2<float> arg must be stripped to its value component.
        bool r_derivs = R.has_derivs()
                        && R.typespec().aggregate() == TypeDesc::SCALAR
                        && R.typespec().simpletype().basetype
                               == TypeDesc::FLOAT;
        if (!r_derivs && A.has_derivs() && !A.typespec().is_triple())
            av += ".val()";
        rop.outputfmt("{}{}", a ? ", " : "", av);
    }
    rop.outputfmt(");\n");
    return true;
}



bool
BackendCpp::sym_carries_derivs(const Symbol& s) const
{
    if (!s.has_derivs())
        return false;
    TypeDesc t = s.typespec().simpletype();
    if (t.basetype != TypeDesc::FLOAT)
        return false;
    return t.aggregate == TypeDesc::SCALAR || t.aggregate == TypeDesc::VEC3;
}



// A float component value for a constructor, stripping a Dual2 scalar to its
// .val().
std::string
BackendCpp::cpp_float_val(const Symbol& s)
{
    std::string v = cpp_value_str(s);
    if (s.has_derivs() && s.typespec().aggregate() == TypeDesc::SCALAR)
        v += ".val()";
    return v;
}



// Build the constructor expression for a triple Result from three float
// component symbols.  When Result carries derivatives (declared Dual2<Vec3> /
// Dual2<Color3>) and any component is itself a Dual2, assemble per-component
// val/dx/dy; otherwise emit a plain 3-arg (optionally Dual2-wrapped) ctor.
std::string
BackendCpp::cpp_triple_ctor(const Symbol& R, const Symbol* c0, const Symbol* c1,
                            const Symbol* c2)
{
    const Symbol* c[3]  = { c0, c1, c2 };
    std::string tn      = lang_sym_type_name(R);  // maybe OSL::Dual2<OSL::Vec3>
    bool rderiv         = sym_carries_derivs(R) && R.typespec().is_triple();
    bool any_comp_deriv = sym_carries_derivs(*c0) || sym_carries_derivs(*c1)
                          || sym_carries_derivs(*c2);

    if (!rderiv) {
        return fmtformat("{}({}, {}, {})", tn, cpp_float_val(*c0),
                         cpp_float_val(*c1), cpp_float_val(*c2));
    }
    // Element type without the Dual2 wrapper, e.g. OSL::Vec3 / OSL::Color3.
    std::string elem = lang_type_name(R.typespec().simpletype());
    if (!any_comp_deriv) {
        // No incoming derivatives: val = elem(c0,c1,c2), dx/dy = 0.
        return fmtformat("{}({}({}, {}, {}))", tn, elem, cpp_float_val(*c0),
                         cpp_float_val(*c1), cpp_float_val(*c2));
    }
    auto part = [&](const char* acc) -> std::string {
        std::string s = elem + "(";
        for (int i = 0; i < 3; ++i) {
            if (i)
                s += ", ";
            if (sym_carries_derivs(*c[i]))
                s += fmtformat("{}.{}()", cpp_value_str(*c[i]), acc);
            else
                s += (std::string(acc) == "val") ? cpp_float_val(*c[i])
                                                 : std::string("0.0f");
        }
        return s + ")";
    };
    return fmtformat("{}({}, {}, {})", tn, part("val"), part("dx"), part("dy"));
}



// Emit `R = Matrix44(diag,0,0,0, 0,diag,0,0, ...)`: a diagonal matrix with the
// scalar `diag` on the diagonal.  Imath's Matrix44(T) sets *every* element, so a
// matrix-from-scalar (construct or assign) must spell out the 16-float ctor.
void
BackendCpp::cpp_emit_matrix_diagonal(const Symbol& R, const std::string& diag)
{
    outputfmt("{}{} = {}(", indentstr(), R.cpp_safe_name(),
              lang_sym_type_name(R));
    for (int i = 0; i < 16; ++i)
        outputfmt("{}{}", i ? ", " : "",
                  ((i % 4) == (i / 4)) ? diag : std::string("0.0f"));
    outputfmt(");\n");
}



// A coordinate-system / colorspace name as a ustringhash_pod, as expected by the
// osl_* transform calls.  String constants are emitted as a uint64_t hash
// (already a pod); string variables are OSL::ustringhash and need .hash().
std::string
BackendCpp::cpp_spacename_pod(const Symbol& s)
{
    std::string v = cpp_value_str(s);
    return s.is_constant() ? v : v + ".hash()";
}



// color (r,g,b) or color ("fromspace", r,g,b): fill the components, then convert
// the named colorspace to RGB in place.  Mirrors llvm_gen_construct_color.
bool
cpp_gen_construct_color(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R        = *rop.opargsym(op, 0);
    bool using_space = (op.nargs() == 5);
    if (!using_space) {
        rop.outputfmtln("{} = {};", R.cpp_safe_name(),
                        rop.cpp_triple_ctor(R, rop.opargsym(op, 1),
                                            rop.opargsym(op, 2),
                                            rop.opargsym(op, 3)));
        return true;
    }
    for (int c = 0; c < 3; ++c)
        rop.outputfmtln("{}[{}] = {};", R.cpp_safe_name(), c,
                        rop.cpp_float_val(*rop.opargsym(op, c + 2)));
    rop.outputfmtln("osl_prepend_color_from((void*)sg, (void*)&{}, {});",
                    R.cpp_safe_name(),
                    rop.cpp_spacename_pod(*rop.opargsym(op, 1)));
    return true;
}



// point/vector/normal (x,y,z), optionally in a named coordinate system: fill the
// components, then transform to common space in place.  Mirrors
// llvm_gen_construct_triple.
bool
cpp_gen_construct_triple(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R        = *rop.opargsym(op, 0);
    bool using_space = (op.nargs() == 5);
    if (!using_space) {
        rop.outputfmtln("{} = {};", R.cpp_safe_name(),
                        rop.cpp_triple_ctor(R, rop.opargsym(op, 1),
                                            rop.opargsym(op, 2),
                                            rop.opargsym(op, 3)));
        return true;
    }
    Symbol& Space  = *rop.opargsym(op, 1);
    std::string rn = R.cpp_safe_name();
    // Build the triple (val + any component derivs), then transform in place.
    // Use the Dual2-aware ctor so deriv-triple results are constructed correctly
    // rather than via R[c] (which a Dual2<Vec3> has no subscript for).
    rop.outputfmtln("{} = {};", rn,
                    rop.cpp_triple_ctor(R, rop.opargsym(op, 2),
                                        rop.opargsym(op, 3),
                                        rop.opargsym(op, 4)));
    // A constant common-space "from" needs no transformation.
    ustring from, to;  // N.B. leave empty for non-constant spaces
    if (Space.is_constant()) {
        from = Space.get_string();
        if (from == Strings::common
            || from == rop.shadingsys().commonspace_synonym())
            return true;
    }
    int vectype = TypeDesc::POINT;
    if (op.opname() == "vector")
        vectype = TypeDesc::VECTOR;
    else if (op.opname() == "normal")
        vectype = TypeDesc::NORMAL;
    int pderiv = rop.sym_carries_derivs(R) ? 1 : 0;
    // The renderer may know of a nonlinear transform for these spaces.
    RendererServices* rend = rop.shadingsys().renderer();
    const char* fn = rend->transform_points(NULL, from, to, 0.0f, NULL, NULL, 0,
                                            (TypeDesc::VECSEMANTICS)vectype)
                         ? "osl_transform_triple_nonlinear"
                         : "osl_transform_triple";
    rop.outputfmtln("{}((void*)sg, (void*)&{}, {}, (void*)&{}, {}, {}, "
                    "OSL::ustring(\"common\").hash(), {});",
                    fn, rn, pderiv, rn, pderiv, rop.cpp_spacename_pod(Space),
                    vectype);
    return true;
}



// matrix constructor.  Forms:
//   matrix (float)                 matrix (space, float)
//   matrix (...16 floats...)       matrix (space, ...16 floats...)
//   matrix (fromspace, tospace)
// Mirrors llvm_gen_matrix.
bool
cpp_gen_matrix(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R             = *rop.opargsym(op, 0);
    int nargs             = op.nargs();
    bool using_space      = (nargs == 3 || nargs == 18);
    bool using_two_spaces = (nargs == 3
                             && rop.opargsym(op, 2)->typespec().is_string());
    int nfloats           = nargs - 1 - (int)using_space;
    std::string rn        = R.cpp_safe_name();
    std::string tn        = rop.lang_sym_type_name(R);

    if (using_two_spaces) {
        rop.outputfmtln("osl_get_from_to_matrix((void*)sg, (void*)&{}, {}, {});",
                        rn, rop.cpp_spacename_pod(*rop.opargsym(op, 1)),
                        rop.cpp_spacename_pod(*rop.opargsym(op, 2)));
        return true;
    }
    if (nfloats == 1) {
        // matrix(f) is a diagonal matrix.
        rop.cpp_emit_matrix_diagonal(
            R, rop.cpp_float_val(*rop.opargsym(op, 1 + (int)using_space)));
    } else {  // nfloats == 16
        rop.outputfmt("{}{} = {}(", rop.indentstr(), rn, tn);
        for (int i = 0; i < 16; ++i)
            rop.outputfmt("{}{}", i ? ", " : "",
                          rop.cpp_float_val(
                              *rop.opargsym(op, i + 1 + (int)using_space)));
        rop.outputfmt(");\n");
    }
    if (using_space)
        rop.outputfmtln("osl_prepend_matrix_from((void*)sg, (void*)&{}, {});",
                        rn, rop.cpp_spacename_pod(*rop.opargsym(op, 1)));
    return true;
}



// Dx/Dy/Dz: extract a partial derivative.  For a Dual2 source these are
// src.dx()/src.dy(); the third partial (Dz) is not stored by Dual2<...,2>, so it
// is zero.  The result itself carries no derivatives.
bool
cpp_gen_DxDy(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));
    bool is_dz      = (op.opname() == "Dz");
    const char* acc = (op.opname() == "Dx") ? "dx" : "dy";
    if (!is_dz && rop.sym_carries_derivs(Src)) {
        rop.outputfmtln("{} = {}.{}();", R.cpp_safe_name(), Src.cpp_safe_name(),
                        acc);
    } else {
        std::string zero
            = R.typespec().is_triple()
                  ? fmtformat("{}(0.0f, 0.0f, 0.0f)",
                              rop.lang_type_name(R.typespec().simpletype()))
                  : std::string("0.0f");
        rop.outputfmtln("{} = {};", R.cpp_safe_name(), zero);
    }
    return true;
}



// int getmatrix (fromspace, tospace, M): osl_get_from_to_matrix(oec, &M, from,
// to) returns the success status into Result.  Mirrors llvm_gen_getmatrix.
bool
cpp_gen_getmatrix(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& From   = *rop.opargsym(op, 1);
    Symbol& To     = *rop.opargsym(op, 2);
    Symbol& M      = *rop.opargsym(op, 3);
    rop.outputfmtln(
        "{} = osl_get_from_to_matrix((void*)sg, (void*)&{}, {}, {});",
        Result.cpp_safe_name(), M.cpp_safe_name(), rop.cpp_spacename_pod(From),
        rop.cpp_spacename_pod(To));
    return true;
}



// transform/transformv/transformn (matrix|fromspace[,tospace], triple p).
// Mirrors llvm_gen_transform.
bool
cpp_gen_transform(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs      = op.nargs();
    Symbol* Result = rop.opargsym(op, 0);
    Symbol* From   = (nargs == 3) ? nullptr : rop.opargsym(op, 1);
    Symbol* To     = rop.opargsym(op, (nargs == 3) ? 1 : 2);
    Symbol* P      = rop.opargsym(op, (nargs == 3) ? 2 : 3);

    // transform(matrix, p): matrix * point — osl_ops has it; use the generic path.
    if (To->typespec().is_matrix())
        return cpp_gen_generic(rop, opnum);

    // Named-space form.  The frontend rewrites the 1-space `transform("to",p)`
    // into 2-space `transform("common","to",p)`, so From is non-null here; the
    // From==nullptr fallback is defensive.
    ustring from, to;  // empty for non-constant spaces
    if ((From == nullptr || From->is_constant()) && To->is_constant()) {
        from        = From ? From->get_string() : Strings::common;
        to          = To->get_string();
        ustring syn = rop.shadingsys().commonspace_synonym();
        if (from == syn)
            from = Strings::common;
        if (to == syn)
            to = Strings::common;
        if (from == to) {
            // Identity transform: just copy P into Result.
            if (Result != P)
                rop.outputfmtln("{} = {};", Result->cpp_safe_name(),
                                rop.cpp_value_str(*P));
            return true;
        }
    }
    int vectype = TypeDesc::POINT;
    if (op.opname() == "transformv")
        vectype = TypeDesc::VECTOR;
    else if (op.opname() == "transformn")
        vectype = TypeDesc::NORMAL;
    RendererServices* rend = rop.shadingsys().renderer();
    const char* fn = rend->transform_points(NULL, from, to, 0.0f, NULL, NULL, 0,
                                            (TypeDesc::VECSEMANTICS)vectype)
                         ? "osl_transform_triple_nonlinear"
                         : "osl_transform_triple";
    std::string from_pod = From
                               ? rop.cpp_spacename_pod(*From)
                               : std::string("OSL::ustring(\"common\").hash()");
    // Pass the real deriv flags so the runtime reads/writes the Dual2<Vec3>
    // storage (and zeroes output derivs when the input carries none).
    rop.outputfmtln("{}((void*)sg, (void*)&{}, {}, (void*)&{}, {}, {}, {}, {});",
                    fn, rop.cpp_value_str(*P),
                    rop.sym_carries_derivs(*P) ? 1 : 0, Result->cpp_safe_name(),
                    rop.sym_carries_derivs(*Result) ? 1 : 0, from_pod,
                    rop.cpp_spacename_pod(*To), vectype);
    return true;
}



// transformc (fromspace, tospace, color p): osl_transformc(oec, &Cin, 0, &Cout,
// 0, from, to).  Mirrors llvm_gen_transformc.
bool
cpp_gen_transformc(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& From   = *rop.opargsym(op, 1);
    Symbol& To     = *rop.opargsym(op, 2);
    Symbol& C      = *rop.opargsym(op, 3);
    rop.outputfmtln(
        "osl_transformc((void*)sg, (void*)&{}, {}, (void*)&{}, {}, {}, {});",
        rop.cpp_value_str(C), rop.sym_carries_derivs(C) ? 1 : 0,
        Result.cpp_safe_name(), rop.sym_carries_derivs(Result) ? 1 : 0,
        rop.cpp_spacename_pod(From), rop.cpp_spacename_pod(To));
    return true;
}



// float luminance (color c): osl_luminance_fv(oec, &result, &color).  The result
// is written through an out-pointer and the call needs the exec context, so it
// can't go through the generic generator.  Mirrors llvm_gen_luminance.
bool
cpp_gen_luminance(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& C(*rop.opargsym(op, 1));
    // Triples carry no derivatives in the C++ backend, so always the _fv form.
    rop.outputfmtln("osl_luminance_fv((void*)sg, (void*)&{}, (void*)&{});",
                    Result.cpp_safe_name(), rop.cpp_value_str(C));
    return true;
}



// float/triple filterwidth(x): osl_filterwidth_fdf(&x) (returns the width) for
// float, osl_filterwidth_vdv(&result, &x) for triple.  The *input* carries the
// derivatives that define the width while the *result* carries none, so the
// generic generator's deriv mangling can't express it — hence a dedicated
// generator.  Mirrors llvm_gen_filterwidth.
bool
cpp_gen_filterwidth(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));
    if (rop.sym_carries_derivs(Src)) {
        if (Src.typespec().is_float()) {
            // Returns a float; assigning to a Dual2 result zeroes its derivs.
            rop.outputfmtln("{} = osl_filterwidth_fdf((void*)&{});",
                            Result.cpp_safe_name(), Src.cpp_safe_name());
        } else {
            // vdv writes only the value (the leading Vec3 == .val()).
            rop.outputfmtln("osl_filterwidth_vdv((void*)&{}, (void*)&{});",
                            Result.cpp_safe_name(), Src.cpp_safe_name());
            // No 2nd-order derivs: zero the result's partials if it carries any.
            if (rop.sym_carries_derivs(Result))
                rop.outputfmtln(
                    "{0}.dx() = OSL::Vec3(0.0f, 0.0f, 0.0f); {0}.dy() = OSL::Vec3(0.0f, 0.0f, 0.0f);",
                    Result.cpp_safe_name());
        }
    } else {
        // No derivatives to be had — result is zero (mirrors llvm_assign_zero).
        std::string zero = Result.typespec().is_triple()
                               ? fmtformat("{}(0.0f, 0.0f, 0.0f)",
                                           rop.lang_type_name(
                                               Result.typespec().simpletype()))
                               : std::string("0.0f");
        rop.outputfmtln("{} = {};", Result.cpp_safe_name(), zero);
    }
    return true;
}



// closure construction:  Result = closure [weight] "name" formal_args... kw...
// Mirrors llvm_gen_closure.  The renderer registry (queried at codegen time)
// supplies the id, struct size, and per-parameter offsets/types/keys.  The
// testshade closures register with no prepare/setup callbacks, so the JIT's
// prepare/setup function-pointer baking is unnecessary: allocate the component,
// zero the parameter memory, memcpy each formal and keyword argument into its
// slot, and store the pointer.  The allocation may return null (zero weight or
// out-of-pool), so all the filling is guarded by `if (comp)`.
bool
cpp_gen_closure(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 2);
    Symbol& Result = *rop.opargsym(op, 0);
    int weighted   = rop.opargsym(op, 1)->typespec().is_string() ? 0 : 1;
    Symbol* weight = weighted ? rop.opargsym(op, 1) : nullptr;
    Symbol& Id     = *rop.opargsym(op, 1 + weighted);
    OSL_DASSERT(Result.typespec().is_closure() && Id.typespec().is_string());
    ustring closure_name = Id.get_string();

    const ClosureRegistry::ClosureEntry* clentry
        = rop.shadingsys().find_closure(closure_name);
    if (!clentry) {
        rop.shadingcontext()->errorfmt(
            "Closure '{}' is not supported by the current renderer, called from {}:{} in shader \"{}\", layer {} \"{}\", group \"{}\"",
            closure_name, op.sourcefile(), op.sourceline(),
            rop.inst()->shadername(), rop.layer(), rop.inst()->layername(),
            rop.group().name());
        return false;
    }

    std::string comp = fmtformat("___clcomp{}", opnum);
    std::string mem  = fmtformat("___clmem{}", opnum);
    if (weighted)
        rop.outputfmtln(
            "void* {} = osl_allocate_weighted_closure_component((void*)sg, {}, {}, (void*)&{});",
            comp, clentry->id, clentry->struct_size,
            rop.cpp_value_str(*weight));
    else
        rop.outputfmtln(
            "void* {} = osl_allocate_closure_component((void*)sg, {}, {});",
            comp, clentry->id, clentry->struct_size);
    rop.outputfmtln("if ({}) {{", comp);
    rop.increment_indent();
    rop.outputfmtln("void* {} = ((OSL::ClosureComponent*){})->data();", mem,
                    comp);
    rop.outputfmtln("std::memset({}, 0, {});", mem, clentry->struct_size);

    // An addressable lvalue for a closure argument.  Aggregate/string/array
    // constants are emitted as named variables (addressable), but a scalar
    // float/int constant is inlined as a literal by cpp_value_str — so
    // materialize a temp for it (can't take the address of an rvalue).
    int argtmp        = 0;
    auto arg_addr_str = [&](Symbol& sym) -> std::string {
        TypeDesc td = sym.typespec().simpletype();
        if (sym.symtype() == SymTypeConst && td.arraylen == 0
            && td.aggregate == 1 && td.basetype != TypeDesc::STRING) {
            std::string t = fmtformat("{}_arg{}", comp, argtmp++);
            rop.outputfmtln("{} {} = {};", rop.lang_type_name(td), t,
                            rop.cpp_value_str(sym));
            return t;
        }
        return rop.cpp_value_str(sym);
    };

    // Formal (positional) parameters: copy each into its registry slot.
    for (int carg = 0; carg < clentry->nformal; ++carg) {
        const ClosureParam& p = clentry->params[carg];
        if (p.key != nullptr)
            break;
        Symbol& sym = *rop.opargsym(op, carg + 2 + weighted);
        rop.outputfmtln("std::memcpy((char*){} + {}, (void*)&{}, {});", mem,
                        p.offset, arg_addr_str(sym), (int)p.type.size());
    }

    // Keyword parameters: (key, value) pairs after the formals; match by
    // name+type against the registry's keyword params (mirrors
    // llvm_gen_keyword_fill).
    int argsoffset = 2 + weighted + clentry->nformal;
    int Nattrs     = (op.nargs() - argsoffset) / 2;
    for (int attr_i = 0; attr_i < Nattrs; ++attr_i) {
        int argno          = attr_i * 2 + argsoffset;
        Symbol& Key        = *rop.opargsym(op, argno);
        Symbol& Value      = *rop.opargsym(op, argno + 1);
        ustring key        = Key.get_string();
        TypeDesc ValueType = Value.typespec().simpletype();
        bool legal         = false;
        for (int t = 0; t < clentry->nkeyword; ++t) {
            const ClosureParam& p = clentry->params[clentry->nformal + t];
            if (equivalent(p.type, ValueType) && !strcmp(key.c_str(), p.key)) {
                rop.outputfmtln("std::memcpy((char*){} + {}, (void*)&{}, {});",
                                mem, p.offset, arg_addr_str(Value),
                                (int)p.type.size());
                legal = true;
                break;
            }
        }
        if (!legal)
            rop.shadingcontext()->warningfmt(
                "Unsupported closure keyword arg \"{}\" for {} ({}:{})", key,
                closure_name, op.sourcefile(), op.sourceline());
    }

    rop.decrement_indent();
    rop.outputfmtln("}}");
    // Store the result last, so `Ci = modifier(Ci)` works.
    rop.outputfmtln("{} = {};", Result.cpp_safe_name(), comp);
    return true;
}



// int raytype(string name): constant name folds to a bit pattern at codegen
// time (osl_raytype_bit); a runtime name dispatches to osl_raytype_name.
// Mirrors llvm_gen_raytype.
bool
cpp_gen_raytype(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Name   = *rop.opargsym(op, 1);
    if (Name.is_constant())
        rop.outputfmtln("{} = osl_raytype_bit((void*)sg, {});",
                        Result.cpp_safe_name(),
                        rop.shadingsys().raytype_bit(Name.get_string()));
    else
        rop.outputfmtln("{} = osl_raytype_name((void*)sg, {});",
                        Result.cpp_safe_name(), rop.cpp_spacename_pod(Name));
    return true;
}



// useparam: the optimizer inserts this pseudo-op right before the point where
// the listed params are used. For each connected param argument, run any
// run_lazily() upstream layer feeding it (guarded by its run-flag, deduped
// within this op) and then load the param's value from GroupData into its
// local. This is where the C++ backend realizes the JIT's lazy layer execution
// (mirrors llvm_gen_useparam -> llvm_run_connected_layers): a run_lazily()
// upstream runs on demand at the point of use, after this layer's own earlier
// ops. A non-lazy upstream already ran via the group entry, so its value is
// already in GroupData and only needs loading.
bool
cpp_gen_useparam(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int this_layer         = rop.layer();
    std::string group_name = rop.group().name().string();
    std::vector<int> already_run;  // upstream layers run for this op
    for (int i = 0; i < op.nargs(); ++i) {
        Symbol& sym  = *rop.opargsym(op, i);
        int symindex = rop.inst()->arg(op.firstarg() + i);
        if (sym.valuesource() != Symbol::ConnectedVal)
            continue;
        bool connected = false;
        for (int c = 0; c < rop.inst()->nconnections(); ++c) {
            const Connection& con = rop.inst()->connection(c);
            if (con.dst.param != symindex)
                continue;
            connected          = true;
            ShaderInstance* up = rop.group()[con.srclayer];
            if (!up->run_lazily()
                || std::find(already_run.begin(), already_run.end(),
                             con.srclayer)
                       != already_run.end())
                continue;
            already_run.push_back(con.srclayer);
            std::string up_func = fmtformat("osl_layer_group_{}_name_{}",
                                            group_name, up->layername());
            rop.outputfmtln("if (!gd->layer_runflags[{}]) {{",
                            rop.layer_remap(con.srclayer));
            rop.increment_indent();
            rop.outputfmtln(
                "{}(sg, gd, userdata_base, output_base, shadeindex, interactive_params);",
                up_func);
            rop.decrement_indent();
            rop.outputfmtln("}}");
        }
        if (!connected)
            continue;
        // Load the (now up-to-date) connected value from GroupData.
        if (sym.typespec().is_array())
            rop.outputfmtln("std::memcpy({}, gd->lay{}param_{}, sizeof({}));",
                            sym.cpp_safe_name(), this_layer,
                            sym.cpp_safe_name(), sym.cpp_safe_name());
        else
            rop.outputfmtln("{} = gd->lay{}param_{};", sym.cpp_safe_name(),
                            this_layer, sym.cpp_safe_name());
    }
    return true;
}



// backfacing / surfacearea: read a scalar ShaderGlobals field directly. The op
// name is the field name (matches ShaderGlobalNameToIndex). Mirrors
// llvm_gen_get_simple_SG_field.
bool
cpp_gen_get_simple_SG_field(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 1);
    Symbol& Result = *rop.opargsym(op, 0);
    rop.outputfmtln("{} = sg->{};", Result.cpp_safe_name(), op.opname());
    return true;
}



// int isconstant(value): folds to a compile-time 0/1 — 1 iff the argument is a
// constant symbol. Mirrors llvm_gen_isconstant.
bool
cpp_gen_isconstant(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    rop.outputfmtln("{} = {};", Result.cpp_safe_name(),
                    A.is_constant() ? 1 : 0);
    return true;
}



// float area(point P): the differential surface area at P, from P's
// derivatives; 0 if P carries none. Mirrors llvm_gen_area. Routed to a
// dedicated generator (not generic) because the generic mangling would emit a
// nonexistent osl_area_fv; the real runtime entry is `float osl_area(void* P)`.
bool
cpp_gen_area(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& P      = *rop.opargsym(op, 1);
    if (!rop.sym_carries_derivs(P)) {
        // No derivatives on P → area is zero (mirrors llvm_assign_zero).
        rop.outputfmtln("{} = 0.0f;", Result.cpp_safe_name());
        return true;
    }
    // osl_area returns the float area; assigning to a Dual2 result zeroes its
    // derivatives (matching the JIT's llvm_zero_derivs).
    rop.outputfmtln("{} = osl_area((void*)&{});", Result.cpp_safe_name(),
                    P.cpp_safe_name());
    return true;
}



// normal calculatenormal(point P): osl_calculatenormal(&Result, sg, &P) using
// P's derivatives. If P carries none, the result is zero. The runtime writes
// only the value, so a deriv-carrying result has its partials zeroed. Mirrors
// llvm_gen_calculatenormal. Routed to a dedicated generator (not generic)
// because the op needs the exec context, which the generic mangling drops.
bool
cpp_gen_calculatenormal(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& P      = *rop.opargsym(op, 1);
    if (!rop.sym_carries_derivs(P)) {
        // No derivatives on P → result is zero (mirrors llvm_assign_zero).
        rop.outputfmtln("{} = {}(0.0f, 0.0f, 0.0f);", Result.cpp_safe_name(),
                        rop.lang_type_name(Result.typespec().simpletype()));
        return true;
    }
    rop.outputfmtln("osl_calculatenormal((void*)&{}, (void*)sg, (void*)&{});",
                    Result.cpp_safe_name(), P.cpp_safe_name());
    // The runtime writes only the value; zero any result partials.
    if (rop.sym_carries_derivs(Result))
        rop.outputfmtln(
            "{0}.dx() = OSL::Vec3(0.0f, 0.0f, 0.0f); {0}.dy() = OSL::Vec3(0.0f, 0.0f, 0.0f);",
            Result.cpp_safe_name());
    return true;
}



// spline/splineinverse(type, value, [knot_count,] knots): builds a mangled
// osl_<op>_<deriv/type codes> name and calls it with out-ptr, the spline-type
// string, value-ptr, knots-ptr, knot count and array length.  Mirrors
// llvm_gen_spline.
bool
cpp_gen_spline(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 4 && op.nargs() <= 5);
    bool has_knot_count = (op.nargs() == 5);
    Symbol& Result      = *rop.opargsym(op, 0);
    Symbol& Spline      = *rop.opargsym(op, 1);
    Symbol& Value       = *rop.opargsym(op, 2);
    Symbol& Knot_count  = *rop.opargsym(op, 3);  // might alias Knots
    Symbol& Knots       = has_knot_count ? *rop.opargsym(op, 4)
                                         : *rop.opargsym(op, 3);

    // Use result derivatives only if the result and an input both carry them.
    bool result_derivs = Result.has_derivs()
                         && (Value.has_derivs() || Knots.has_derivs());
    std::string name = fmtformat("osl_{}_", op.opname());
    if (result_derivs)
        name += "d";
    if (Result.typespec().is_float())
        name += "f";
    else if (Result.typespec().is_triple())
        name += "v";
    if (result_derivs && Value.has_derivs())
        name += "d";
    if (Value.typespec().is_float())
        name += "f";
    else if (Value.typespec().is_triple())
        name += "v";
    if (result_derivs && Knots.has_derivs())
        name += "d";
    if (Knots.typespec().simpletype().elementtype() == TypeDesc::FLOAT)
        name += "f";
    else if (Knots.typespec().simpletype().elementtype().aggregate
             == TypeDesc::VEC3)
        name += "v";

    std::string knotcount
        = has_knot_count ? rop.cpp_value_str(Knot_count)
                         : fmtformat("{}", Knots.typespec().arraylength());

    // The x value is passed by void*; a constant x has no address, so
    // materialize it into a temp (cpp_void_ptr_arg handles const scalars).
    std::string valarg = rop.cpp_void_ptr_arg(Value,
                                              fmtformat("___splx{}", opnum));

    // The knots are passed by void* to the runtime. The cpp backend declares a
    // deriv-carrying knot array AoS (Dual2<elem>[N]), but the runtime reads a
    // plain array whose layout depends on the chosen variant: a deriv-knot
    // variant expects the SoA deriv layout [val[N]][dx[N]][dy[N]]; a non-deriv
    // variant reads just [val[N]]. Either way the AoS Dual2 array is the wrong
    // layout, so build a matching shadow. (A non-deriv knot array is already a
    // plain value array in the expected layout — pass it directly.)
    bool knot_derivs_in_call = result_derivs && Knots.has_derivs();
    std::string knotarg      = fmtformat("(void*)&{}", Knots.cpp_safe_name());
    if (rop.sym_carries_derivs(Knots)) {
        int n           = Knots.typespec().arraylength();
        std::string elt = rop.lang_type_name(Knots.typespec().simpletype());
        std::string sh  = fmtformat("___splk{}", opnum);
        std::string k   = Knots.cpp_safe_name();
        if (knot_derivs_in_call) {
            rop.outputfmtln("{} {}[{}];", elt, sh, 3 * n);
            rop.outputfmtln(
                "for (int ___i = 0; ___i < {0}; ++___i) {{ {1}[___i] = {2}[___i].val(); {1}[{0}+___i] = {2}[___i].dx(); {1}[2*{0}+___i] = {2}[___i].dy(); }}",
                n, sh, k);
        } else {
            rop.outputfmtln("{} {}[{}];", elt, sh, n);
            rop.outputfmtln(
                "for (int ___i = 0; ___i < {0}; ++___i) {1}[___i] = {2}[___i].val();",
                n, sh, k);
        }
        knotarg = fmtformat("(void*){}", sh);
    }
    rop.outputfmtln("{}((void*)&{}, {}, {}, {}, {}, {});", name,
                    Result.cpp_safe_name(), rop.cpp_spacename_pod(Spline),
                    valarg, knotarg, knotcount, Knots.typespec().arraylength());

    // Result wants derivs but none propagated: zero them (the non-deriv runtime
    // variant wrote only the value into the Dual2 storage).
    if (Result.has_derivs() && !result_derivs
        && rop.sym_carries_derivs(Result)) {
        if (Result.typespec().is_triple())
            rop.outputfmtln(
                "{0}.dx() = OSL::Vec3(0.0f, 0.0f, 0.0f); {0}.dy() = OSL::Vec3(0.0f, 0.0f, 0.0f);",
                Result.cpp_safe_name());
        else
            rop.outputfmtln("{0}.dx() = 0.0f; {0}.dy() = 0.0f;",
                            Result.cpp_safe_name());
    }
    return true;
}



// Scalar arg passed by value to a runtime function: strip a Dual2 to its value.
std::string
BackendCpp::cpp_scalar_val(const Symbol& s)
{
    std::string str = cpp_value_str(s);
    if (sym_carries_derivs(s))
        str += ".val()";
    return str;
}



// pointcloud_search (filename, center, radius, max_points, [sort,] attrs...):
// the "index"/"distance" attributes map to dedicated out-args; every other
// (name,value) pair is pushed into a names/types/values arena via
// osl_pointcloud_write_helper and fetched by the runtime. Mirrors
// llvm_gen_pointcloud_search.
//
// The cpp backend stores a deriv-carrying output array AoS (Dual2<elem>[N]),
// but the runtime writes a contiguous value layout: a value-only array for a
// regular attribute, and a [val][dx][dy] SoA region for the distances when the
// center carries derivatives (derivs_offset = N). For any deriv-carrying output
// array we therefore allocate a matching plain shadow, pass that to the call,
// and scatter the result back into the Dual2 array afterward (zeroing the
// element derivs unless the SoA path supplied them).
bool
cpp_gen_pointcloud_search(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 5);
    Symbol& Result     = *rop.opargsym(op, 0);
    Symbol& Filename   = *rop.opargsym(op, 1);
    Symbol& Center     = *rop.opargsym(op, 2);
    Symbol& Radius     = *rop.opargsym(op, 3);
    Symbol& Max_points = *rop.opargsym(op, 4);

    int attr_arg_offset = 5;
    Symbol* Sort        = nullptr;
    if (op.nargs() > 5 && rop.opargsym(op, 5)->typespec().is_int()) {
        Sort = rop.opargsym(op, 5);
        ++attr_arg_offset;
    }
    int nattrs = (op.nargs() - attr_arg_offset) / 2;

    // Attribute arena (one slot per regular attribute).
    std::string names  = fmtformat("___pcs_names{}", opnum);
    std::string types  = fmtformat("___pcs_types{}", opnum);
    std::string values = fmtformat("___pcs_values{}", opnum);
    int asz            = nattrs > 0 ? nattrs : 1;
    rop.outputfmtln("OSL::ustringhash {}[{}];", names, asz);
    rop.outputfmtln("OSL::TypeDesc {}[{}];", types, asz);
    rop.outputfmtln("void* {}[{}];", values, asz);

    // Deferred scatter of a value shadow back into an AoS Dual2 array.
    struct Pending {
        std::string arr, sh;
        bool soa_distance;
    };
    std::vector<Pending> pending;

    std::string indices_expr;
    std::string distances_expr = "(void*)nullptr";
    int derivs_offset          = 0;
    int extra_attrs            = 0;
    int capacity               = 0x7FFFFFFF;
    bool have_indices          = false;

    for (int i = 0; i < nattrs; ++i) {
        Symbol& Name        = *rop.opargsym(op, attr_arg_offset + i * 2);
        Symbol& Value       = *rop.opargsym(op, attr_arg_offset + i * 2 + 1);
        TypeDesc simpletype = Value.typespec().simpletype();
        int N               = (int)simpletype.numelements();
        if (Name.is_constant() && Name.get_string() == "index"
            && simpletype.elementtype() == TypeDesc::INT) {
            indices_expr = fmtformat("(void*)&{}", Value.cpp_safe_name());
            have_indices = true;
        } else if (Name.is_constant() && Name.get_string() == "distance"
                   && simpletype.elementtype() == TypeDesc::FLOAT) {
            if (rop.sym_carries_derivs(Value)) {
                std::string sh = fmtformat("___pcs_dist{}", opnum);
                if (Center.has_derivs()) {
                    rop.outputfmtln("float {}[{}];", sh, 3 * N);
                    derivs_offset = N;
                } else {
                    rop.outputfmtln("float {}[{}];", sh, N > 0 ? N : 1);
                }
                distances_expr = fmtformat("(void*){}", sh);
                pending.push_back(
                    { Value.cpp_safe_name(), sh, Center.has_derivs() });
            } else {
                distances_expr = fmtformat("(void*)&{}", Value.cpp_safe_name());
            }
        } else {
            // Regular attribute: arena slot + (for a deriv array) a value shadow.
            long long tdp = OSL::bitcast<long long, TypeDesc>(simpletype);
            std::string valptr;
            if (rop.sym_carries_derivs(Value)) {
                std::string elt = rop.lang_type_name(simpletype);
                std::string sh = fmtformat("___pcs_a{}_{}", opnum, extra_attrs);
                rop.outputfmtln("{} {}[{}];", elt, sh, N > 0 ? N : 1);
                valptr = fmtformat("(void*){}", sh);
                pending.push_back({ Value.cpp_safe_name(), sh, false });
            } else {
                valptr = fmtformat("(void*)&{}", Value.cpp_safe_name());
            }
            rop.outputfmtln(
                "osl_pointcloud_write_helper((void*){}, (void*){}, (void*){}, {}, {}, {}LL, {});",
                names, types, values, extra_attrs, rop.cpp_spacename_pod(Name),
                tdp, valptr);
            ++extra_attrs;
        }
        capacity = std::min(N, capacity);
    }

    // No caller-supplied index array: allocate one sized to the arrays' capacity.
    if (!have_indices) {
        std::string idx = fmtformat("___pcs_idx{}", opnum);
        rop.outputfmtln("int {}[{}];", idx, capacity > 0 ? capacity : 1);
        indices_expr = fmtformat("(void*){}", idx);
    }

    // max_points clamped to the arrays' capacity (per the OSL spec, results are
    // limited to what the output arrays can hold).
    std::string maxp;
    if (Max_points.is_constant()) {
        int cmax = Max_points.get_int();
        if (capacity < cmax) {
            rop.shadingcontext()->warningfmt(
                "Arrays too small for pointcloud lookup at ({}:{})",
                op.sourcefile(), op.sourceline());
            maxp = fmtformat("{}", capacity);
        } else {
            maxp = rop.cpp_scalar_val(Max_points);
        }
    } else {
        std::string mp = rop.cpp_scalar_val(Max_points);
        maxp = fmtformat("({0} <= ({1}) ? {0} : ({1}))", capacity, mp);
    }

    std::string sort = Sort ? rop.cpp_scalar_val(*Sort) : std::string("0");
    rop.outputfmtln(
        "{} = osl_pointcloud_search((void*)sg, {}, (void*)&{}, {}, {}, {}, {}, {}, {}, {}, (void*){}, (void*){}, (void*){});",
        Result.cpp_safe_name(), rop.cpp_spacename_pod(Filename),
        Center.cpp_safe_name(), rop.cpp_scalar_val(Radius), maxp, sort,
        indices_expr, distances_expr, derivs_offset, extra_attrs, names, types,
        values);

    // Scatter shadows back into the Dual2 arrays (bounded by the found count).
    for (auto& p : pending) {
        if (p.soa_distance)
            rop.outputfmtln(
                "for (int ___i = 0; ___i < {0}; ++___i) {{ {1}[___i].val() = {2}[___i]; {1}[___i].dx() = {2}[{3}+___i]; {1}[___i].dy() = {2}[2*{3}+___i]; }}",
                Result.cpp_safe_name(), p.arr, p.sh, derivs_offset);
        else
            rop.outputfmtln(
                "for (int ___i = 0; ___i < {0}; ++___i) {{ {1}[___i].val() = {2}[___i]; {1}[___i].clear_d(); }}",
                Result.cpp_safe_name(), p.arr, p.sh);
    }
    return true;
}



// pointcloud_get (filename, indices, count, attr_name, data): fetch one
// attribute for the given indices. Count is clamped to the smaller of the
// indices/data array lengths. A deriv-carrying data array uses a value shadow
// (the runtime writes only values; the derivs are then zeroed). Mirrors
// llvm_gen_pointcloud_get.
bool
cpp_gen_pointcloud_get(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 6);
    Symbol& Result    = *rop.opargsym(op, 0);
    Symbol& Filename  = *rop.opargsym(op, 1);
    Symbol& Indices   = *rop.opargsym(op, 2);
    Symbol& Count     = *rop.opargsym(op, 3);
    Symbol& Attr_name = *rop.opargsym(op, 4);
    Symbol& Data      = *rop.opargsym(op, 5);

    int element_count  = std::min(Data.typespec().arraylength(),
                                  Indices.typespec().arraylength());
    std::string nclamp = fmtformat("___pcg_n{}", opnum);
    rop.outputfmtln("int {0} = ({1} <= ({2}) ? {1} : ({2}));", nclamp,
                    element_count, rop.cpp_scalar_val(Count));

    long long tdp = OSL::bitcast<long long, TypeDesc>(
        Data.typespec().simpletype());
    bool data_shadow = rop.sym_carries_derivs(Data);
    std::string sh, dataarg;
    if (data_shadow) {
        std::string elt = rop.lang_type_name(Data.typespec().simpletype());
        int N           = Data.typespec().arraylength();
        sh              = fmtformat("___pcg_sh{}", opnum);
        rop.outputfmtln("{} {}[{}];", elt, sh, N > 0 ? N : 1);
        dataarg = fmtformat("(void*){}", sh);
    } else {
        dataarg = fmtformat("(void*)&{}", Data.cpp_safe_name());
    }
    rop.outputfmtln(
        "{} = osl_pointcloud_get((void*)sg, {}, (void*)&{}, {}, {}, {}LL, {});",
        Result.cpp_safe_name(), rop.cpp_spacename_pod(Filename),
        Indices.cpp_safe_name(), nclamp, rop.cpp_spacename_pod(Attr_name), tdp,
        dataarg);
    if (data_shadow)
        rop.outputfmtln(
            "for (int ___i = 0; ___i < {0}; ++___i) {{ {1}[___i].val() = {2}[___i]; {1}[___i].clear_d(); }}",
            nclamp, Data.cpp_safe_name(), sh);
    return true;
}



// pointcloud_write (filename, position, attrs...): store a point with its
// attributes. Each (name,value) pair is written into a names/types/values arena
// (read by value, so a Dual2 attribute's leading value is what's stored).
// Mirrors llvm_gen_pointcloud_write.
bool
cpp_gen_pointcloud_write(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 3);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& Pos      = *rop.opargsym(op, 2);
    int nattrs       = (op.nargs() - 3) / 2;

    std::string names  = fmtformat("___pcw_names{}", opnum);
    std::string types  = fmtformat("___pcw_types{}", opnum);
    std::string values = fmtformat("___pcw_values{}", opnum);
    int asz            = nattrs > 0 ? nattrs : 1;
    rop.outputfmtln("OSL::ustringhash {}[{}];", names, asz);
    rop.outputfmtln("OSL::TypeDesc {}[{}];", types, asz);
    rop.outputfmtln("void* {}[{}];", values, asz);

    for (int i = 0; i < nattrs; ++i) {
        Symbol& Name  = *rop.opargsym(op, 3 + 2 * i);
        Symbol& Value = *rop.opargsym(op, 3 + 2 * i + 1);
        long long tdp = OSL::bitcast<long long, TypeDesc>(
            Value.typespec().simpletype());
        rop.outputfmtln(
            "osl_pointcloud_write_helper((void*){}, (void*){}, (void*){}, {}, {}, {}LL, (void*)&{});",
            names, types, values, i, rop.cpp_spacename_pod(Name), tdp,
            Value.cpp_safe_name());
    }
    rop.outputfmtln(
        "{} = osl_pointcloud_write((void*)sg, {}, (void*)&{}, {}, (void*){}, (void*){}, (void*){});",
        Result.cpp_safe_name(), rop.cpp_spacename_pod(Filename),
        Pos.cpp_safe_name(), nattrs, names, types, values);
    return true;
}



// dict_find(string|int source, string query) -> int node id. Two variants by
// source type (osl_dict_find_iss / _iis). Mirrors llvm_gen_dict_find. Dedicated
// (not generic) because the dict ops take the exec context, which the generic
// mangling drops.
bool
cpp_gen_dict_find(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Source   = *rop.opargsym(op, 1);
    Symbol& Query    = *rop.opargsym(op, 2);
    bool sourceint   = Source.typespec().is_int();
    const char* func = sourceint ? "osl_dict_find_iis" : "osl_dict_find_iss";
    std::string src  = sourceint ? rop.cpp_value_str(Source)
                                 : rop.cpp_spacename_pod(Source);
    rop.outputfmtln("{} = {}((void*)sg, {}, {});", Result.cpp_safe_name(), func,
                    src, rop.cpp_spacename_pod(Query));
    return true;
}



// int dict_value(int nodeID, string name, output TYPE value): writes the
// attribute into value, returns whether found. Mirrors llvm_gen_dict_value.
bool
cpp_gen_dict_value(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& NodeID = *rop.opargsym(op, 1);
    Symbol& Name   = *rop.opargsym(op, 2);
    Symbol& Value  = *rop.opargsym(op, 3);
    long long tdp  = OSL::bitcast<long long, TypeDesc>(
        Value.typespec().simpletype());
    rop.outputfmtln("{} = osl_dict_value((void*)sg, {}, {}, {}LL, (void*)&{});",
                    Result.cpp_safe_name(), rop.cpp_value_str(NodeID),
                    rop.cpp_spacename_pod(Name), tdp, Value.cpp_safe_name());
    return true;
}



// int dict_next(int nodeID): advance to the next matching node. Mirrors
// llvm_gen_dict_next.
bool
cpp_gen_dict_next(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& NodeID = *rop.opargsym(op, 1);
    rop.outputfmtln("{} = osl_dict_next((void*)sg, {});",
                    Result.cpp_safe_name(), rop.cpp_value_str(NodeID));
    return true;
}



// getattribute (eight flavors: optional object name, optional array index).
// Emits the common osl_get_attribute() call (the build_attribute_getter spec
// path is an OptiX/rs-bitcode optimization not used by the C++ DSO path).
// Mirrors the non-spec branch of llvm_gen_getattribute.
bool
cpp_gen_getattribute(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs          = op.nargs();
    bool array_lookup  = rop.opargsym(op, nargs - 2)->typespec().is_int();
    bool object_lookup = rop.opargsym(op, 2)->typespec().is_string()
                         && nargs >= 4;
    int object_slot = (int)object_lookup;
    int attrib_slot = object_slot + 1;
    int index_slot  = array_lookup ? nargs - 2 : 0;

    Symbol& Result      = *rop.opargsym(op, 0);
    Symbol& ObjectName  = *rop.opargsym(op, object_slot);
    Symbol& Attribute   = *rop.opargsym(op, attrib_slot);
    Symbol& Index       = *rop.opargsym(op, index_slot);
    Symbol& Destination = *rop.opargsym(op, nargs - 1);

    // The destination's type is passed through to the renderer; pack it into the
    // long long the runtime bit-casts back to a TypeDesc (TYPEDESC macro).
    TypeDesc dest_type  = Destination.typespec().simpletype();
    long long tdpacked  = OSL::bitcast<long long, TypeDesc>(dest_type);
    std::string objname = object_lookup ? rop.cpp_spacename_pod(ObjectName)
                                        : std::string("OSL::ustring().hash()");
    std::string idx     = array_lookup ? rop.cpp_value_str(Index)
                                       : std::string("0");
    rop.outputfmtln(
        "{} = osl_get_attribute((void*)sg, {}, {}, {}, {}, {}, {}LL, (void*)&{});",
        Result.cpp_safe_name(), Destination.has_derivs() ? 1 : 0, objname,
        rop.cpp_spacename_pod(Attribute), array_lookup ? 1 : 0, idx, tdpacked,
        Destination.cpp_safe_name());
    return true;
}



// Pack a message/attribute data symbol's type into the long long the runtime
// bit-casts to a TypeDesc.  Closures use TypeDesc(UNKNOWN, arraylen) per the
// JIT "secret handshake".
static long long
cpp_message_type_packed(const Symbol& Data)
{
    TypeDesc td = Data.typespec().is_closure_based()
                      ? TypeDesc(TypeDesc::UNKNOWN,
                                 Data.typespec().arraylength())
                      : Data.typespec().simpletype();
    return OSL::bitcast<long long, TypeDesc>(td);
}



// A "(void*)&storage" expression for sym, materializing a temp first when sym
// is an inlined scalar constant (int/float consts have no address — they're
// spelled as literals by cpp_value_str; strings/aggregates/arrays are
// declared variables and can be addressed directly).
std::string
BackendCpp::cpp_void_ptr_arg(const Symbol& sym, const std::string& tmpname)
{
    TypeDesc td = sym.typespec().simpletype();
    if (sym.symtype() == SymTypeConst && td.arraylen == 0 && td.aggregate == 1
        && td.basetype != TypeDesc::STRING) {
        outputfmtln("{} {} = {};", lang_type_name(td), tmpname,
                    cpp_value_str(sym));
        return "(void*)&" + tmpname;
    }
    return "(void*)&" + sym.cpp_safe_name();
}



// setmessage(name, data): osl_setmessage(sg, name, type, &data, layerid,
// sourcefile, sourceline).  Mirrors llvm_gen_setmessage.
bool
cpp_gen_setmessage(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Name = *rop.opargsym(op, 0);
    Symbol& Data = *rop.opargsym(op, 1);
    std::string dataptr
        = rop.cpp_void_ptr_arg(Data, fmtformat("___msgdata{}", opnum));
    rop.outputfmtln(
        "osl_setmessage((OSL::ShaderGlobals*)sg, {}, {}LL, {}, {}, {}ULL, {});",
        rop.cpp_spacename_pod(Name), cpp_message_type_packed(Data), dataptr,
        rop.inst()->id(), (uint64_t)op.sourcefile().hash(), op.sourceline());
    return true;
}



// getmessage([source,] name, data): a constant source of "trace" reads from the
// trace result (osl_trace_get); otherwise osl_getmessage.  Mirrors
// llvm_gen_getmessage.
bool
cpp_gen_getmessage(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int has_source = (op.nargs() == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Source = *rop.opargsym(op, 1);
    Symbol& Name   = *rop.opargsym(op, 1 + has_source);
    Symbol& Data   = *rop.opargsym(op, 2 + has_source);

    if (has_source && Source.is_constant() && Source.get_string() == "trace") {
        rop.outputfmtln(
            "{} = osl_trace_get((void*)sg, {}, {}LL, (void*)&{}, {});",
            Result.cpp_safe_name(), rop.cpp_spacename_pod(Name),
            cpp_message_type_packed(Data), Data.cpp_safe_name(),
            Data.has_derivs() ? 1 : 0);
        return true;
    }
    std::string source = has_source ? rop.cpp_spacename_pod(Source)
                                    : std::string("OSL::ustring().hash()");
    rop.outputfmtln(
        "{} = osl_getmessage((OSL::ShaderGlobals*)sg, {}, {}, {}LL, (void*)&{}, {}, {}, {}ULL, {});",
        Result.cpp_safe_name(), source, rop.cpp_spacename_pod(Name),
        cpp_message_type_packed(Data), Data.cpp_safe_name(),
        Data.has_derivs() ? 1 : 0, rop.inst()->id(),
        (unsigned long long)op.sourcefile().hash(), op.sourceline());
    return true;
}



// color blackbody(float tempK) / color wavelength_color(float nm): both call
// osl_<op>_vf(sg, &result, temp); result derivs are punted to zero.  Mirrors
// llvm_gen_blackbody.
bool
cpp_gen_blackbody(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Temperature(*rop.opargsym(op, 1));
    std::string temp = rop.cpp_value_str(Temperature);
    if (rop.sym_carries_derivs(Temperature))
        temp += ".val()";
    rop.outputfmtln("osl_{}_vf((void*)sg, (void*)&{}, {});", op.opname(),
                    Result.cpp_safe_name(), temp);
    if (rop.sym_carries_derivs(Result))
        rop.outputfmtln(
            "{0}.dx() = OSL::Vec3(0.0f, 0.0f, 0.0f); {0}.dy() = OSL::Vec3(0.0f, 0.0f, 0.0f);",
            Result.cpp_safe_name());
    return true;
}



// Parse trace() optional (token,value) args and emit osl_trace_set_* calls into
// the named TraceOpt.  Mirrors llvm_gen_trace_options.
static void
cpp_gen_trace_options(BackendCpp& rop, int opnum, int first_optional_arg,
                      const std::string& optvar)
{
    rop.outputfmtln("OSL::TraceOpt {};", optvar);
    rop.outputfmtln("osl_init_trace_options((void*)sg, (void*)&{});", optvar);
    Opcode& op(rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg; a + 1 < op.nargs(); a += 2) {
        Symbol& Name(*rop.opargsym(op, a));
        Symbol& Val(*rop.opargsym(op, a + 1));
        if (!Name.typespec().is_string())
            break;
        ustring name     = Name.get_string();
        TypeDesc valtype = Val.typespec().simpletype();
        std::string v    = rop.cpp_value_str(Val);
        if (name == Strings::mindist && valtype == TypeDesc::FLOAT)
            rop.outputfmtln("osl_trace_set_mindist((void*)&{}, {});", optvar,
                            v);
        else if (name == Strings::maxdist && valtype == TypeDesc::FLOAT)
            rop.outputfmtln("osl_trace_set_maxdist((void*)&{}, {});", optvar,
                            v);
        else if (name == Strings::shade && valtype == TypeDesc::INT)
            rop.outputfmtln("osl_trace_set_shade((void*)&{}, {});", optvar, v);
        else if (name == Strings::traceset && valtype == TypeDesc::STRING)
            rop.outputfmtln("osl_trace_set_traceset((void*)&{}, {});", optvar,
                            rop.cpp_spacename_pod(Val));
        else
            rop.shadingcontext()->errorfmt(
                "Unknown trace() optional argument: \"{}\" ({}:{})", name,
                op.sourcefile(), op.sourceline());
    }
}



// int trace(point pos, vector dir, ...): osl_trace(sg, &opt, &pos.val/dx/dy,
// &dir.val/dx/dy).  Mirrors llvm_gen_trace.
bool
cpp_gen_trace(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result     = *rop.opargsym(op, 0);
    Symbol& Pos        = *rop.opargsym(op, 1);
    Symbol& Dir        = *rop.opargsym(op, 2);
    std::string optvar = fmtformat("_traceopt{}", opnum);
    cpp_gen_trace_options(rop, opnum, 3, optvar);

    // Block d (0=val,1=dx,2=dy) of a triple: a deriv-carrying triple stores
    // val/dx/dy contiguously, so address each; otherwise all three point at the
    // value (matching llvm_get_pointer clamping a non-deriv symbol).
    auto block = [&](const Symbol& S, int d) -> std::string {
        if (rop.sym_carries_derivs(S)) {
            const char* acc = d == 0 ? "val" : d == 1 ? "dx" : "dy";
            return fmtformat("(void*)&{}.{}()", S.cpp_safe_name(), acc);
        }
        return fmtformat("(void*)&{}", S.cpp_safe_name());
    };
    rop.outputfmtln(
        "{} = osl_trace((void*)sg, (void*)&{}, {}, {}, {}, {}, {}, {});",
        Result.cpp_safe_name(), optvar, block(Pos, 0), block(Pos, 1),
        block(Pos, 2), block(Dir, 0), block(Dir, 1), block(Dir, 2));
    rop.inst()->has_trace_op(true);
    return true;
}



// int regex_match/regex_search(string subject, [int results[],] string pat):
// osl_regex_impl(sg, subject, &results, nresults, pattern, fullmatch).  Mirrors
// llvm_gen_regex.
bool
cpp_gen_regex(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs             = op.nargs();
    Symbol& Result        = *rop.opargsym(op, 0);
    Symbol& Subject       = *rop.opargsym(op, 1);
    bool do_match_results = (nargs == 4);
    bool fullmatch        = (op.opname() == "regex_match");
    Symbol& Match         = *rop.opargsym(op, 2);
    Symbol& Pattern       = *rop.opargsym(op, 2 + do_match_results);
    std::string results = do_match_results ? "(void*)&" + Match.cpp_safe_name()
                                           : std::string("nullptr");
    int nresults        = do_match_results ? Match.typespec().arraylength() : 0;
    rop.outputfmtln("{} = osl_regex_impl((void*)sg, {}, {}, {}, {}, {});",
                    Result.cpp_safe_name(), rop.cpp_spacename_pod(Subject),
                    results, nresults, rop.cpp_spacename_pod(Pattern),
                    fullmatch ? 1 : 0);
    return true;
}



// int split(string str, output string results[], [string sep, [int maxsplit]]):
// osl_split(str, results, sep, maxsplit, resultslen).  Mirrors llvm_gen_split.
bool
cpp_gen_split(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R            = *rop.opargsym(op, 0);
    Symbol& Str          = *rop.opargsym(op, 1);
    Symbol& Results      = *rop.opargsym(op, 2);
    int arraylen         = Results.typespec().arraylength();
    std::string sep      = (op.nargs() >= 4)
                               ? rop.cpp_spacename_pod(*rop.opargsym(op, 3))
                               : std::string("OSL::ustring(\"\").hash()");
    std::string maxsplit = (op.nargs() >= 5)
                               ? rop.cpp_value_str(*rop.opargsym(op, 4))
                               : fmtformat("{}", arraylen);
    rop.outputfmtln("{} = osl_split({}, (OSL::ustringhash_pod*){}, {}, {}, {});",
                    R.cpp_safe_name(), rop.cpp_spacename_pod(Str),
                    Results.cpp_safe_name(), sep, maxsplit, arraylen);
    return true;
}



// select(a, b, cond): per-component `cond ? b : a` (cond != 0).  A scalar cond
// applies to every component; a triple cond is per-component.  Inline, no
// runtime call.  Mirrors llvm_gen_select (incl. derivative propagation).
bool
cpp_gen_select(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& A(*rop.opargsym(op, 1));
    Symbol& B(*rop.opargsym(op, 2));
    Symbol& X(*rop.opargsym(op, 3));
    bool triple = R.typespec().is_triple();
    bool rdual  = rop.sym_carries_derivs(R);
    int xc      = X.typespec().aggregate();

    // Component ci, deriv level d (0=val,1=dx,2=dy) of a symbol; a non-deriv
    // symbol contributes 0 for d>0 (matching the JIT's zeroed-deriv branch).
    auto acc = [&](const Symbol& s, int ci, int d) -> std::string {
        std::string v = rop.cpp_value_str(s);
        bool sd       = rop.sym_carries_derivs(s);
        bool st       = s.typespec().is_triple();
        if (d == 0) {
            if (sd)
                v += ".val()";
        } else if (!sd)
            return std::string("0.0f");
        else
            v += (d == 1) ? ".dx()" : ".dy()";
        if (st)
            v += fmtformat("[{}]", ci);
        return v;
    };
    auto cond = [&](int i) -> std::string {
        int xi        = (i >= xc) ? 0 : i;
        std::string v = rop.cpp_value_str(X);
        if (rop.sym_carries_derivs(X))
            v += ".val()";
        if (X.typespec().is_triple())
            v += fmtformat("[{}]", xi);
        return fmtformat("({} != 0)", v);
    };
    auto sel = [&](int i, int d) -> std::string {
        return fmtformat("({} ? {} : {})", cond(i), acc(B, i, d), acc(A, i, d));
    };

    std::string tn = rop.lang_type_name(R.typespec().simpletype());
    if (!triple) {
        if (rdual)
            rop.outputfmtln("{} = OSL::Dual2<float>({}, {}, {});",
                            R.cpp_safe_name(), sel(0, 0), sel(0, 1), sel(0, 2));
        else
            rop.outputfmtln("{} = {};", R.cpp_safe_name(), sel(0, 0));
    } else {
        auto vec = [&](int d) {
            return fmtformat("{}({}, {}, {})", tn, sel(0, d), sel(1, d),
                             sel(2, d));
        };
        if (rdual) {
            rop.outputfmtln("{}.val() = {};", R.cpp_safe_name(), vec(0));
            rop.outputfmtln("{}.dx() = {};", R.cpp_safe_name(), vec(1));
            rop.outputfmtln("{}.dy() = {};", R.cpp_safe_name(), vec(2));
        } else {
            rop.outputfmtln("{} = {};", R.cpp_safe_name(), vec(0));
        }
    }
    return true;
}



// Comparison ops (eq, neq, lt, le, gt, ge).  OSL compares aggregates
// component-wise and broadcasts a scalar against a triple/matrix; the result is
// the AND of the per-component comparisons (OR for !=).  Mirrors
// llvm_gen_compare_op.
bool
cpp_gen_compare_op(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& A(*rop.opargsym(op, 1));
    Symbol& B(*rop.opargsym(op, 2));
    ustring opn = op.opname();

    // closure == 0 / closure != 0
    if (A.typespec().is_closure_based()) {
        rop.outputfmtln("{} = ({} {} nullptr);", R.cpp_safe_name(),
                        rop.cpp_value_str(A), opn == "eq" ? "==" : "!=");
        return true;
    }

    const char* o = (opn == "eq")    ? "=="
                    : (opn == "neq") ? "!="
                    : (opn == "lt")  ? "<"
                    : (opn == "le")  ? "<="
                    : (opn == "gt")  ? ">"
                                     : ">=";

    // String eq/neq: string consts are raw uint64 hashes; wrap them as
    // ustringhash so the comparison is well-typed.
    if (A.typespec().is_string()) {
        auto s = [&](const Symbol& x) -> std::string {
            std::string v = rop.cpp_value_str(x);
            return x.is_constant()
                       ? fmtformat("OSL::ustringhash::from_hash({})", v)
                       : v;
        };
        rop.outputfmtln("{} = ({} {} {});", R.cpp_safe_name(), s(A), o, s(B));
        return true;
    }

    int nc     = std::max(A.typespec().aggregate(), B.typespec().aggregate());
    bool a_mat = A.typespec().is_matrix();
    bool b_mat = B.typespec().is_matrix();
    const char* combine = (opn == "neq") ? " || " : " && ";

    // Component i of a symbol; a scalar broadcasts, except off-diagonal entries
    // compared against a matrix are taken as 0 (matrix-vs-scalar trickery).
    auto comp = [&](const Symbol& s, int i,
                    bool other_is_matrix) -> std::string {
        std::string v = rop.cpp_value_str(s);
        if (rop.sym_carries_derivs(s))
            v += ".val()";
        TypeSpec t = s.typespec();
        if (t.is_matrix())
            return fmtformat("{}[{}][{}]", v, i / 4, i % 4);
        if (t.is_triple())
            return fmtformat("{}[{}]", v, i);
        if (other_is_matrix && (i / 4) != (i % 4))
            return std::string("0.0f");
        return v;
    };

    std::string expr;
    for (int i = 0; i < nc; ++i) {
        if (i)
            expr += combine;
        expr += fmtformat("({} {} {})", comp(A, i, b_mat), o,
                          comp(B, i, a_mat));
    }
    rop.outputfmtln("{} = ({});", R.cpp_safe_name(), expr);
    return true;
}



// Helper: parse texture optional args and emit osl_texture_set_* calls.
// Fills in alpha/errormessage pointer expressions (as strings) for later use.
// Returns false if parsing fails (shouldn't happen after optimization).
struct CppTexOptResult {
    std::string alpha_ptr;  // "(void*)&var" or "nullptr"
    std::string dalphadx_ptr;
    std::string dalphady_ptr;
    std::string errormsg_ptr;
};

static CppTexOptResult
cpp_gen_texture_options(BackendCpp& rop, int opnum, int first_optional_arg,
                        bool tex3d, int nchans, const std::string& optvar)
{
    CppTexOptResult r;
    r.alpha_ptr = r.dalphadx_ptr = r.dalphady_ptr = r.errormsg_ptr = "nullptr";

    Opcode& op(rop.inst()->ops()[opnum]);
    bool missingcolor_arena = false;

    for (int a = first_optional_arg; a < op.nargs(); ++a) {
        Symbol& Name(*rop.opargsym(op, a));
        if (!Name.typespec().is_string() || !Name.is_constant())
            break;
        if (++a >= op.nargs())
            break;
        Symbol& Val(*rop.opargsym(op, a));
        TypeDesc valtype = Val.typespec().simpletype();
        ustring name     = Name.get_string();

        // Produce a C++ expression for Val suitable for a float/int argument.
        auto val_f = [&]() -> std::string {
            if (Val.is_constant() && valtype == TypeDesc::INT)
                return fmtformat("(float){}", Val.get_int());
            return fmtformat("(float)({})", rop.cpp_value_str(Val));
        };
        auto val_i = [&]() -> std::string { return rop.cpp_value_str(Val); };

        if ((name == Strings::width || name == Strings::blur)
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            const char* stfn = (name == Strings::width)
                                   ? "osl_texture_set_stwidth"
                                   : "osl_texture_set_stblur";
            const char* rfn  = (name == Strings::width)
                                   ? "osl_texture_set_rwidth"
                                   : "osl_texture_set_rblur";
            rop.outputfmtln("{}((void*)&{}, {});", stfn, optvar, val_f());
            if (tex3d)
                rop.outputfmtln("{}((void*)&{}, {});", rfn, optvar, val_f());
            continue;
        }
        if (name == Strings::swidth
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            rop.outputfmtln("osl_texture_set_swidth((void*)&{}, {});", optvar,
                            val_f());
            continue;
        }
        if (name == Strings::twidth
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            rop.outputfmtln("osl_texture_set_twidth((void*)&{}, {});", optvar,
                            val_f());
            continue;
        }
        if (name == Strings::rwidth
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            rop.outputfmtln("osl_texture_set_rwidth((void*)&{}, {});", optvar,
                            val_f());
            continue;
        }
        if (name == Strings::sblur
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            rop.outputfmtln("osl_texture_set_sblur((void*)&{}, {});", optvar,
                            val_f());
            continue;
        }
        if (name == Strings::tblur
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            rop.outputfmtln("osl_texture_set_tblur((void*)&{}, {});", optvar,
                            val_f());
            continue;
        }
        if (name == Strings::rblur
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            rop.outputfmtln("osl_texture_set_rblur((void*)&{}, {});", optvar,
                            val_f());
            continue;
        }
        if (name == Strings::fill
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            rop.outputfmtln("osl_texture_set_fill((void*)&{}, {});", optvar,
                            val_f());
            continue;
        }
        if (name == Strings::firstchannel && valtype == TypeDesc::INT) {
            rop.outputfmtln("osl_texture_set_firstchannel((void*)&{}, {});",
                            optvar, val_i());
            continue;
        }
        if (name == Strings::subimage && valtype == TypeDesc::INT) {
            rop.outputfmtln("osl_texture_set_subimage((void*)&{}, {});", optvar,
                            val_i());
            continue;
        }
        if (name == Strings::subimage && valtype == TypeDesc::STRING) {
            if (Val.is_constant() && Val.get_string().empty())
                continue;
            rop.outputfmtln("osl_texture_set_subimagename((void*)&{}, {});",
                            optvar, rop.cpp_spacename_pod(Val));
            continue;
        }
        if (name == Strings::wrap && valtype == TypeDesc::STRING) {
            if (Val.is_constant()) {
                int code = (int)OIIO::TextureOpt::decode_wrapmode(
                    Val.get_string());
                rop.outputfmtln("osl_texture_set_stwrap_code((void*)&{}, {});",
                                optvar, code);
                if (tex3d)
                    rop.outputfmtln(
                        "osl_texture_set_rwrap_code((void*)&{}, {});", optvar,
                        code);
            } else {
                rop.outputfmtln("osl_texture_set_stwrap((void*)&{}, {});",
                                optvar, rop.cpp_spacename_pod(Val));
                if (tex3d)
                    rop.outputfmtln("osl_texture_set_rwrap((void*)&{}, {});",
                                    optvar, rop.cpp_spacename_pod(Val));
            }
            continue;
        }
        if (name == Strings::swrap && valtype == TypeDesc::STRING) {
            if (Val.is_constant()) {
                int code = (int)OIIO::TextureOpt::decode_wrapmode(
                    Val.get_string());
                rop.outputfmtln("osl_texture_set_swrap_code((void*)&{}, {});",
                                optvar, code);
            } else {
                rop.outputfmtln("osl_texture_set_swrap((void*)&{}, {});",
                                optvar, rop.cpp_spacename_pod(Val));
            }
            continue;
        }
        if (name == Strings::twrap && valtype == TypeDesc::STRING) {
            if (Val.is_constant()) {
                int code = (int)OIIO::TextureOpt::decode_wrapmode(
                    Val.get_string());
                rop.outputfmtln("osl_texture_set_twrap_code((void*)&{}, {});",
                                optvar, code);
            } else {
                rop.outputfmtln("osl_texture_set_twrap((void*)&{}, {});",
                                optvar, rop.cpp_spacename_pod(Val));
            }
            continue;
        }
        if (name == Strings::rwrap && valtype == TypeDesc::STRING) {
            if (Val.is_constant()) {
                int code = (int)OIIO::TextureOpt::decode_wrapmode(
                    Val.get_string());
                rop.outputfmtln("osl_texture_set_rwrap_code((void*)&{}, {});",
                                optvar, code);
            } else {
                rop.outputfmtln("osl_texture_set_rwrap((void*)&{}, {});",
                                optvar, rop.cpp_spacename_pod(Val));
            }
            continue;
        }
        if (name == Strings::interp && valtype == TypeDesc::STRING) {
            if (Val.is_constant()) {
                int code = tex_interp_to_code(Val.get_string());
                if (code >= 0)
                    rop.outputfmtln(
                        "osl_texture_set_interp_code((void*)&{}, {});", optvar,
                        code);
            } else {
                rop.outputfmtln("osl_texture_set_interp((void*)&{}, {});",
                                optvar, rop.cpp_spacename_pod(Val));
            }
            continue;
        }
        if (name == Strings::alpha && valtype == TypeDesc::FLOAT) {
            r.alpha_ptr = fmtformat("(void*)&{}", Val.cpp_safe_name());
            if (rop.sym_carries_derivs(Val)) {
                r.dalphadx_ptr = fmtformat("(void*)&{}.dx()",
                                           Val.cpp_safe_name());
                r.dalphady_ptr = fmtformat("(void*)&{}.dy()",
                                           Val.cpp_safe_name());
            }
            continue;
        }
        if (name == Strings::errormessage && valtype == TypeDesc::STRING) {
            r.errormsg_ptr = fmtformat("(void*)&{}", Val.cpp_safe_name());
            continue;
        }
        if (name == Strings::missingcolor
            && equivalent(valtype, OIIO::TypeColor)) {
            if (!missingcolor_arena) {
                rop.outputfmtln("float _missing[4] = {{}};");
                rop.outputfmtln(
                    "osl_texture_set_missingcolor_arena((void*)&{}, (void*)_missing);",
                    optvar);
                missingcolor_arena = true;
            }
            rop.outputfmtln(
                "std::memcpy(_missing, (void*)&{}, 3*sizeof(float));",
                rop.cpp_value_str(Val));
            continue;
        }
        if (name == Strings::missingalpha && valtype == TypeDesc::FLOAT) {
            if (!missingcolor_arena) {
                rop.outputfmtln("float _missing[4] = {{}};");
                rop.outputfmtln(
                    "osl_texture_set_missingcolor_arena((void*)&{}, (void*)_missing);",
                    optvar);
                missingcolor_arena = true;
            }
            rop.outputfmtln(
                "osl_texture_set_missingcolor_alpha((void*)&{}, {}, {});",
                optvar, nchans, val_f());
            continue;
        }
        // colorspace and time: accept and ignore (like JIT)
        if (name == Strings::colorspace || name == Strings::time)
            continue;
        // Unknown option — emit a comment and skip
        rop.outputfmtln("// UNIMPLEMENTED texture option: {}", name);
    }
    return r;
}



// texture(filename, s, t, ...) — 2D texture lookup.  Mirrors llvm_gen_texture.
bool
cpp_gen_texture(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& S        = *rop.opargsym(op, 2);
    Symbol& T        = *rop.opargsym(op, 3);
    int nchans       = Result.typespec().aggregate();

    bool user_derivs       = false;
    int first_optional_arg = 4;
    if (op.nargs() > 4 && rop.opargsym(op, 4)->typespec().is_float()) {
        user_derivs        = true;
        first_optional_arg = 8;
    }

    rop.outputfmtln("{{ // texture");
    rop.increment_indent();

    // TextureOpt
    rop.outputfmtln("OIIO::TextureOpt _tex_opt;");
    rop.outputfmtln("osl_init_texture_options((void*)sg, (void*)&_tex_opt);");

    CppTexOptResult toi = cpp_gen_texture_options(rop, opnum,
                                                  first_optional_arg, false,
                                                  nchans, "_tex_opt");

    // Coordinate values and derivatives
    std::string s_val, t_val, dsdx, dtdx, dsdy, dtdy;
    if (user_derivs) {
        s_val = rop.sym_carries_derivs(S)
                    ? fmtformat("{}.val()", rop.cpp_value_str(S))
                    : rop.cpp_value_str(S);
        t_val = rop.sym_carries_derivs(T)
                    ? fmtformat("{}.val()", rop.cpp_value_str(T))
                    : rop.cpp_value_str(T);
        Symbol& Dsdx(*rop.opargsym(op, 4));
        Symbol& Dtdx(*rop.opargsym(op, 5));
        Symbol& Dsdy(*rop.opargsym(op, 6));
        Symbol& Dtdy(*rop.opargsym(op, 7));
        dsdx = rop.sym_carries_derivs(Dsdx)
                   ? fmtformat("{}.val()", rop.cpp_value_str(Dsdx))
                   : rop.cpp_value_str(Dsdx);
        dtdx = rop.sym_carries_derivs(Dtdx)
                   ? fmtformat("{}.val()", rop.cpp_value_str(Dtdx))
                   : rop.cpp_value_str(Dtdx);
        dsdy = rop.sym_carries_derivs(Dsdy)
                   ? fmtformat("{}.val()", rop.cpp_value_str(Dsdy))
                   : rop.cpp_value_str(Dsdy);
        dtdy = rop.sym_carries_derivs(Dtdy)
                   ? fmtformat("{}.val()", rop.cpp_value_str(Dtdy))
                   : rop.cpp_value_str(Dtdy);
    } else if (rop.sym_carries_derivs(S)) {
        s_val = fmtformat("{}.val()", rop.cpp_value_str(S));
        t_val = fmtformat("{}.val()", rop.cpp_value_str(T));
        dsdx  = fmtformat("{}.dx()", rop.cpp_value_str(S));
        dtdx  = fmtformat("{}.dx()", rop.cpp_value_str(T));
        dsdy  = fmtformat("{}.dy()", rop.cpp_value_str(S));
        dtdy  = fmtformat("{}.dy()", rop.cpp_value_str(T));
    } else {
        s_val = rop.cpp_value_str(S);
        t_val = rop.cpp_value_str(T);
        dsdx = dtdx = dsdy = dtdy = "0.0f";
    }

    // Result pointers
    std::string res_ptr, resdx_ptr, resdy_ptr;
    if (rop.sym_carries_derivs(Result)) {
        res_ptr   = fmtformat("(void*)&{}.val()", Result.cpp_safe_name());
        resdx_ptr = fmtformat("(void*)&{}.dx()", Result.cpp_safe_name());
        resdy_ptr = fmtformat("(void*)&{}.dy()", Result.cpp_safe_name());
    } else {
        res_ptr   = fmtformat("(void*)&{}", Result.cpp_safe_name());
        resdx_ptr = resdy_ptr = "nullptr";
    }

    rop.outputfmtln("osl_texture((void*)sg, {}, nullptr, (void*)&_tex_opt,",
                    rop.cpp_spacename_pod(Filename));
    rop.outputfmtln("    {}, {}, {}, {}, {}, {},", s_val, t_val, dsdx, dtdx,
                    dsdy, dtdy);
    rop.outputfmtln("    {}, {}, {}, {},", nchans, res_ptr, resdx_ptr,
                    resdy_ptr);
    rop.outputfmtln("    {}, {}, {}, {});", toi.alpha_ptr, toi.dalphadx_ptr,
                    toi.dalphady_ptr, toi.errormsg_ptr);

    rop.decrement_indent();
    rop.outputfmtln("}} // texture");
    return true;
}



// texture3d(filename, P, ...) — 3D texture lookup.  Mirrors llvm_gen_texture3d.
bool
cpp_gen_texture3d(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& P        = *rop.opargsym(op, 2);
    int nchans       = Result.typespec().aggregate();

    bool user_derivs       = false;
    int first_optional_arg = 3;
    if (op.nargs() > 3 && rop.opargsym(op, 3)->typespec().is_triple()) {
        user_derivs        = true;
        first_optional_arg = 6;
    }

    rop.outputfmtln("{{ // texture3d");
    rop.increment_indent();

    rop.outputfmtln("OIIO::TextureOpt _tex_opt;");
    rop.outputfmtln("osl_init_texture_options((void*)sg, (void*)&_tex_opt);");

    CppTexOptResult toi = cpp_gen_texture_options(rop, opnum,
                                                  first_optional_arg, true,
                                                  nchans, "_tex_opt");

    // P and its derivatives
    std::string p_ptr, dpdx_ptr, dpdy_ptr, dpdz_ptr;
    if (user_derivs) {
        Symbol& Dpdx(*rop.opargsym(op, 3));
        Symbol& Dpdy(*rop.opargsym(op, 4));
        Symbol& Dpdz(*rop.opargsym(op, 5));
        // For user derivs, pass the val() portion of each (or direct if not dual)
        auto triple_val_ptr = [&](Symbol& sym) -> std::string {
            if (rop.sym_carries_derivs(sym))
                return fmtformat("(void*)&{}.val()", sym.cpp_safe_name());
            return fmtformat("(void*)&{}", sym.cpp_safe_name());
        };
        p_ptr    = triple_val_ptr(P);
        dpdx_ptr = triple_val_ptr(Dpdx);
        dpdy_ptr = triple_val_ptr(Dpdy);
        dpdz_ptr = triple_val_ptr(Dpdz);
    } else if (rop.sym_carries_derivs(P)) {
        p_ptr    = fmtformat("(void*)&{}.val()", P.cpp_safe_name());
        dpdx_ptr = fmtformat("(void*)&{}.dx()", P.cpp_safe_name());
        dpdy_ptr = fmtformat("(void*)&{}.dy()", P.cpp_safe_name());
        dpdz_ptr = "nullptr";
    } else {
        // Emit a local zero Vec3 for the missing derivatives
        rop.outputfmtln("OSL::Vec3 _zero3(0.0f, 0.0f, 0.0f);");
        p_ptr    = fmtformat("(void*)&{}", P.cpp_safe_name());
        dpdx_ptr = dpdy_ptr = dpdz_ptr = "(void*)&_zero3";
    }

    std::string res_ptr, resdx_ptr, resdy_ptr;
    if (rop.sym_carries_derivs(Result)) {
        res_ptr   = fmtformat("(void*)&{}.val()", Result.cpp_safe_name());
        resdx_ptr = fmtformat("(void*)&{}.dx()", Result.cpp_safe_name());
        resdy_ptr = fmtformat("(void*)&{}.dy()", Result.cpp_safe_name());
    } else {
        res_ptr   = fmtformat("(void*)&{}", Result.cpp_safe_name());
        resdx_ptr = resdy_ptr = "nullptr";
    }

    rop.outputfmtln("osl_texture3d((void*)sg, {}, nullptr, (void*)&_tex_opt,",
                    rop.cpp_spacename_pod(Filename));
    rop.outputfmtln("    {}, {}, {}, {},", p_ptr, dpdx_ptr, dpdy_ptr, dpdz_ptr);
    rop.outputfmtln("    {}, {}, {}, {},", nchans, res_ptr, resdx_ptr,
                    resdy_ptr);
    rop.outputfmtln("    {}, {}, {}, {});", toi.alpha_ptr, toi.dalphadx_ptr,
                    toi.dalphady_ptr, toi.errormsg_ptr);

    rop.decrement_indent();
    rop.outputfmtln("}} // texture3d");
    return true;
}



// environment(filename, R, ...) — environment map lookup.
// Mirrors llvm_gen_environment.
bool
cpp_gen_environment(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& R        = *rop.opargsym(op, 2);
    int nchans       = Result.typespec().aggregate();

    bool user_derivs       = false;
    int first_optional_arg = 3;
    if (op.nargs() > 3 && rop.opargsym(op, 3)->typespec().is_triple()) {
        user_derivs        = true;
        first_optional_arg = 5;
    }

    rop.outputfmtln("{{ // environment");
    rop.increment_indent();

    rop.outputfmtln("OIIO::TextureOpt _tex_opt;");
    rop.outputfmtln("osl_init_texture_options((void*)sg, (void*)&_tex_opt);");

    CppTexOptResult toi = cpp_gen_texture_options(rop, opnum,
                                                  first_optional_arg, false,
                                                  nchans, "_tex_opt");

    std::string r_ptr, drdx_ptr, drdy_ptr;
    if (user_derivs) {
        Symbol& Drdx(*rop.opargsym(op, 3));
        Symbol& Drdy(*rop.opargsym(op, 4));
        auto triple_val_ptr = [&](Symbol& sym) -> std::string {
            if (rop.sym_carries_derivs(sym))
                return fmtformat("(void*)&{}.val()", sym.cpp_safe_name());
            return fmtformat("(void*)&{}", sym.cpp_safe_name());
        };
        r_ptr    = triple_val_ptr(R);
        drdx_ptr = triple_val_ptr(Drdx);
        drdy_ptr = triple_val_ptr(Drdy);
    } else if (rop.sym_carries_derivs(R)) {
        r_ptr    = fmtformat("(void*)&{}.val()", R.cpp_safe_name());
        drdx_ptr = fmtformat("(void*)&{}.dx()", R.cpp_safe_name());
        drdy_ptr = fmtformat("(void*)&{}.dy()", R.cpp_safe_name());
    } else {
        rop.outputfmtln("OSL::Vec3 _zero3(0.0f, 0.0f, 0.0f);");
        r_ptr    = fmtformat("(void*)&{}", R.cpp_safe_name());
        drdx_ptr = drdy_ptr = "(void*)&_zero3";
    }

    std::string res_ptr, resdx_ptr, resdy_ptr;
    if (rop.sym_carries_derivs(Result)) {
        res_ptr   = fmtformat("(void*)&{}.val()", Result.cpp_safe_name());
        resdx_ptr = fmtformat("(void*)&{}.dx()", Result.cpp_safe_name());
        resdy_ptr = fmtformat("(void*)&{}.dy()", Result.cpp_safe_name());
    } else {
        res_ptr   = fmtformat("(void*)&{}", Result.cpp_safe_name());
        resdx_ptr = resdy_ptr = "nullptr";
    }

    rop.outputfmtln("osl_environment((void*)sg, {}, nullptr, (void*)&_tex_opt,",
                    rop.cpp_spacename_pod(Filename));
    rop.outputfmtln("    {}, {}, {},", r_ptr, drdx_ptr, drdy_ptr);
    rop.outputfmtln("    {}, {}, {}, {},", nchans, res_ptr, resdx_ptr,
                    resdy_ptr);
    rop.outputfmtln("    {}, {}, {}, {});", toi.alpha_ptr, toi.dalphadx_ptr,
                    toi.dalphady_ptr, toi.errormsg_ptr);

    rop.decrement_indent();
    rop.outputfmtln("}} // environment");
    return true;
}



// gettextureinfo(filename, dataname, data) or
// gettextureinfo(filename, s, t, dataname, data).
// Mirrors llvm_gen_gettextureinfo.
bool
cpp_gen_gettextureinfo(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4 || op.nargs() == 6);
    bool use_coords  = (op.nargs() == 6);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& Dataname = *rop.opargsym(op, use_coords ? 4 : 2);
    Symbol& Data     = *rop.opargsym(op, use_coords ? 5 : 3);

    TypeDesc dattype = Data.typespec().simpletype();
    if (use_coords) {
        Symbol& S(*rop.opargsym(op, 2));
        Symbol& T(*rop.opargsym(op, 3));
        std::string s_val = rop.sym_carries_derivs(S)
                                ? fmtformat("{}.val()", rop.cpp_value_str(S))
                                : rop.cpp_value_str(S);
        std::string t_val = rop.sym_carries_derivs(T)
                                ? fmtformat("{}.val()", rop.cpp_value_str(T))
                                : rop.cpp_value_str(T);
        rop.outputfmtln(
            "{} = osl_get_textureinfo_st((void*)sg, {}, nullptr, {}, {}, {}, {}, {}, {}, (void*)&{}, nullptr);",
            Result.cpp_safe_name(), rop.cpp_spacename_pod(Filename), s_val,
            t_val, rop.cpp_spacename_pod(Dataname), (int)dattype.basetype,
            (int)dattype.arraylen, (int)dattype.aggregate,
            Data.cpp_safe_name());
    } else {
        rop.outputfmtln(
            "{} = osl_get_textureinfo((void*)sg, {}, nullptr, {}, {}, {}, {}, (void*)&{}, nullptr);",
            Result.cpp_safe_name(), rop.cpp_spacename_pod(Filename),
            rop.cpp_spacename_pod(Dataname), (int)dattype.basetype,
            (int)dattype.arraylen, (int)dattype.aggregate,
            Data.cpp_safe_name());
    }
    return true;
}



// unary ops
bool
cpp_gen_unary_op(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    const char* opsym = "UNKNOWN";
    if (op.opname() == "neg")
        opsym = "-";
    else if (op.opname() == "compl")
        opsym = "~";

    else
        OSL_ASSERT_MSG(0, "Unknown unary op %s", op.opname().c_str());
    rop.outputfmtln("{} = {} {};", R.cpp_safe_name(), opsym,
                    rop.cpp_value_str(A));
    return true;
}



// binary ops
bool
cpp_gen_binary_op(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));

    // Closure arithmetic: add of two closures, or mul of a closure by a
    // float/color weight.  Closures are pointers, so these are runtime calls,
    // not C++ operators.  Mirrors llvm_gen_add / llvm_gen_mul.
    if (R.typespec().is_closure()) {
        if (op.opname() == "add") {
            rop.outputfmtln("{} = osl_add_closure_closure((void*)sg, {}, {});",
                            R.cpp_safe_name(), rop.cpp_value_str(A),
                            rop.cpp_value_str(B));
        } else {  // mul: one operand is the closure, the other the weight
            Symbol& Cl = A.typespec().is_closure() ? A : B;
            Symbol& W  = A.typespec().is_closure() ? B : A;
            if (W.typespec().is_float())
                rop.outputfmtln("{} = osl_mul_closure_float((void*)sg, {}, {});",
                                R.cpp_safe_name(), rop.cpp_value_str(Cl),
                                rop.cpp_value_str(W));
            else
                rop.outputfmtln(
                    "{} = osl_mul_closure_color((void*)sg, {}, (void*)&{});",
                    R.cpp_safe_name(), rop.cpp_value_str(Cl),
                    rop.cpp_value_str(W));
        }
        return true;
    }

    const char* opsym   = "UNKNOWN";
    bool scalar_promote = false;  // true for ops that need triple broadcast
    if (op.opname() == "add") {
        opsym          = "+";
        scalar_promote = true;
    } else if (op.opname() == "sub") {
        opsym          = "-";
        scalar_promote = true;
    } else if (op.opname() == "mul")
        opsym = "*";

    else if (op.opname() == "eq")
        opsym = "==";
    else if (op.opname() == "neq")
        opsym = "!=";
    else if (op.opname() == "lt")
        opsym = "<";
    else if (op.opname() == "gt")
        opsym = ">";
    else if (op.opname() == "le")
        opsym = "<=";
    else if (op.opname() == "ge")
        opsym = ">=";

    else if (op.opname() == "bitand")
        opsym = "&";
    else if (op.opname() == "bitor")
        opsym = "|";
    else if (op.opname() == "xor")
        opsym = "^";
    else if (op.opname() == "shl")
        opsym = "<<";
    else if (op.opname() == "shr")
        opsym = ">>";

    else if (op.opname() == "and")
        opsym = "&&";
    else if (op.opname() == "or")
        opsym = "||";

    else
        OSL_ASSERT_MSG(0, "Unknown binary op %s", op.opname().c_str());

    // String eq/neq: string vars are OSL::ustringhash, but string constants are
    // static const uint64_t (no implicit conversion).  For string comparisons,
    // wrap uint64_t constants in OSL::ustringhash::from_hash() so both sides
    // match.  Non-const string variables don't need wrapping (already ustringhash).
    if ((op.opname() == "eq" || op.opname() == "neq")
        && A.typespec().is_string()) {
        auto str_cmp_expr = [&](const Symbol& s) -> std::string {
            std::string v = rop.cpp_value_str(s);
            if (s.is_constant())
                return fmtformat("OSL::ustringhash::from_hash({})", v);
            return v;
        };
        rop.outputfmtln("{} = {} {} {};", R.cpp_safe_name(), str_cmp_expr(A),
                        opsym, str_cmp_expr(B));
        return true;
    }

    // When the result does not carry derivatives but a scalar operand is Dual2,
    // extract .val() so the Dual2 → scalar assignment compiles.  This also
    // drops the derivative on that path, matching LLVM behavior when the result
    // symbol has has_derivs() == false.  A deriv-carrying result (scalar OR
    // triple) keeps the operand derivs: the Dual2 operator*/+/- chain rule
    // (dual.h) yields a Dual2 result, so e.g. Dual2<Vec3> * Dual2<float> stays
    // deriv-correct.  `force_strip` is used by the scalar-broadcast path below,
    // which feeds operands into a triple constructor that has no Dual2 overload.
    bool r_dual     = rop.sym_carries_derivs(R);
    auto scalar_str = [&](const Symbol& s, bool force_strip) -> std::string {
        std::string str = rop.cpp_value_str(s);
        // Strip .val() when the result carries no derivs but the operand does.
        // Applies to both scalar Dual2<float> and triple Dual2<Vec3/Color3>.
        if ((force_strip || !r_dual) && s.has_derivs())
            str += ".val()";
        return str;
    };

    // add/sub with one triple and one scalar: Color3/Vec3 have no +/- with
    // scalar, so broadcast the scalar to triple via 3-arg constructor first.
    if (scalar_promote
        && A.typespec().is_triple() != B.typespec().is_triple()) {
        std::string tn = rop.lang_sym_type_name(R);
        if (A.typespec().is_triple()) {
            std::string bv = scalar_str(B, /*force_strip=*/true);
            rop.outputfmtln("{} = {} {} {}({}, {}, {});", R.cpp_safe_name(),
                            rop.cpp_value_str(A), opsym, tn, bv, bv, bv);
        } else {
            std::string av = scalar_str(A, /*force_strip=*/true);
            rop.outputfmtln("{} = {}({}, {}, {}) {} {};", R.cpp_safe_name(), tn,
                            av, av, av, opsym, rop.cpp_value_str(B));
        }
        return true;
    }
    rop.outputfmtln("{} = {} {} {};", R.cpp_safe_name(),
                    scalar_str(A, /*force_strip=*/false), opsym,
                    scalar_str(B, /*force_strip=*/false));
    return true;
}



// C++ code generator for 'div'.
//
// Dispatch strategy:
//   matrix   → osl_div_mmm / osl_div_mmf / osl_div_mfm  (no derivs)
//   int      → osl_div_iii  (safe integer divide)
//   triple   → osl_div_vvv / osl_div_vvf / osl_div_vfv   (void* ABI)
//   float    → C++ division using OSL::Dual2 operators when any deriv is
//              needed; osl_div_fff (safe divide) otherwise.
//              When result has no derivs but an arg does, extract .val() so
//              the derivative is intentionally dropped (matches LLVM behavior).
bool
cpp_gen_div(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));

    if (R.typespec().is_matrix()) {
        if (A.typespec().is_float())
            rop.outputfmtln("osl_div_mfm((void*)&{}, {}, (void*)&{});",
                            R.cpp_safe_name(), rop.cpp_value_str(A),
                            B.cpp_safe_name());
        else if (B.typespec().is_float())
            rop.outputfmtln("osl_div_mmf((void*)&{}, (void*)&{}, {});",
                            R.cpp_safe_name(), A.cpp_safe_name(),
                            rop.cpp_value_str(B));
        else
            rop.outputfmtln("osl_div_mmm((void*)&{}, (void*)&{}, (void*)&{});",
                            R.cpp_safe_name(), A.cpp_safe_name(),
                            B.cpp_safe_name());
        return true;
    }

    if (R.typespec().is_int()) {
        rop.outputfmtln("{} = osl_div_iii({}, {});", R.cpp_safe_name(),
                        rop.cpp_value_str(A), rop.cpp_value_str(B));
        return true;
    }

    if (R.typespec().is_triple()) {
        bool a_triple  = A.typespec().is_triple();
        bool b_triple  = B.typespec().is_triple();
        std::string rv = "(void*)&" + R.cpp_safe_name();
        std::string av = a_triple ? "(void*)&" + A.cpp_safe_name()
                                  : rop.cpp_value_str(A);
        std::string bv = b_triple ? "(void*)&" + B.cpp_safe_name()
                                  : rop.cpp_value_str(B);
        const char* fn = (a_triple && b_triple) ? "osl_div_vvv"
                         : a_triple             ? "osl_div_vvf"
                                                : "osl_div_vfv";
        rop.outputfmtln("{}({}, {}, {});", fn, rv, av, bv);
        return true;
    }

    // Float scalar — handle OSL::Dual2 derivative propagation.
    bool r_dual = R.has_derivs();
    bool a_dual = A.has_derivs() && !A.typespec().is_triple();
    bool b_dual = B.has_derivs() && !B.typespec().is_triple();

    if (r_dual) {
        // Result carries derivatives: safe-divide with Dual2 propagation,
        // matching llvm_gen_div (raw Dual2 operator/ would NaN/Inf on a zero
        // divisor instead of flushing to 0).  Wrap each operand to Dual2<float>
        // so a plain-float operand promotes with zero derivatives.
        rop.outputfmtln(
            "{} = osl_div_dual(OSL::Dual2<float>({}), OSL::Dual2<float>({}));",
            R.cpp_safe_name(), rop.cpp_value_str(A), rop.cpp_value_str(B));
    } else {
        // Result does not need derivatives; strip Dual2 to plain value.
        std::string av = rop.cpp_value_str(A);
        std::string bv = rop.cpp_value_str(B);
        if (a_dual)
            av += ".val()";
        if (b_dual)
            bv += ".val()";
        rop.outputfmtln("{} = osl_div_fff({}, {});", R.cpp_safe_name(), av, bv);
    }
    return true;
}



// C++ code generator for loop ops: for, while, dowhile.
//
// All three use a while(true)/break structure to avoid goto for the common
// case.  'break' always emits the natural C++ 'break' keyword.  'continue'
// emits the natural 'continue' keyword for 'while' loops (where it correctly
// re-evaluates the condition); for 'for'/'dowhile' loops whose body contains a
// 'continue' op we emit a single step-label and a goto so the step/cond ops
// still run before the next iteration.
bool
cpp_gen_loop_op(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& cond   = *rop.opargsym(op, 0);
    ustring opname = op.opname();

    // Decide whether the body has a 'continue' that needs a step-label goto.
    // 'while' never needs it: natural 'continue' goes to the top of the loop
    // which re-runs cond_ops — exactly right.  For 'for'/'dowhile', a plain
    // 'continue' would skip the step/cond ops, so we need a label.
    bool need_step_label = (opname != Strings::op_while)
                           && rop.body_has_continue(op.jump(1), op.jump(2));
    std::string step_lbl;
    if (need_step_label)
        step_lbl = fmtformat("cpp_loop_step_{}", rop.new_loop_label_id());

    // Push context: empty string = emit natural 'continue;' keyword.
    rop.push_loop_context(step_lbl);

    if (opname == Strings::op_for) {
        // Init ops, then while(true){ cond; if(!cond) break; body; [step_lbl:] step; }
        rop.build_cpp_code(opnum + 1, op.jump(0), false);
        rop.outputfmtln("while (true) {{");
        rop.increment_indent();
        rop.build_cpp_code(op.jump(0), op.jump(1), false);
        rop.outputfmtln("if (!{}) break;", cond.cpp_safe_name());
        rop.build_cpp_code(op.jump(1), op.jump(2), false);
        if (!step_lbl.empty())
            rop.outputfmtln("{}:;", step_lbl);
        rop.build_cpp_code(op.jump(2), op.jump(3), false);
        rop.decrement_indent();
        rop.outputfmtln("}}");
    } else if (opname == Strings::op_while) {
        // while(true){ cond_ops; if(!cond) break; body; }
        rop.outputfmtln("while (true) {{");
        rop.increment_indent();
        rop.build_cpp_code(op.jump(0), op.jump(1), false);
        rop.outputfmtln("if (!{}) break;", cond.cpp_safe_name());
        rop.build_cpp_code(op.jump(1), op.jump(2), false);
        rop.decrement_indent();
        rop.outputfmtln("}}");
    } else {  // dowhile
        // do{ body; [step_lbl:] step; cond_ops; } while(cond);
        rop.outputfmtln("do {{");
        rop.increment_indent();
        rop.build_cpp_code(op.jump(1), op.jump(2), false);
        if (!step_lbl.empty())
            rop.outputfmtln("{}:;", step_lbl);
        rop.build_cpp_code(op.jump(2), op.jump(3), false);
        rop.build_cpp_code(op.jump(0), op.jump(1), false);
        rop.decrement_indent();
        rop.outputfmtln("}} while ({});", cond.cpp_safe_name());
    }

    rop.pop_loop_context();
    return true;
}



// Exported wrappers for the printf-family ops, called by generated shader DSOs.
// rs_printfmt and friends have hidden visibility and cannot be bound by
// generated DSOs; these wrappers route through the exported RendererServices
// virtual methods so any renderer implementation is reached correctly.

extern "C" OSL_DLL_EXPORT void
osl_cpp_printfmt(void* sg_void, uint64_t fmt_hash, int32_t arg_count,
                 const uint8_t* etypes, uint32_t values_size,
                 const uint8_t* values)
{
    auto* sg = reinterpret_cast<ShaderGlobals*>(sg_void);
    sg->renderer->printfmt(sg, ustringhash(fmt_hash), arg_count,
                           reinterpret_cast<const EncodedType*>(etypes),
                           values_size, const_cast<uint8_t*>(values));
}



extern "C" OSL_DLL_EXPORT void
osl_cpp_errorfmt(void* sg_void, uint64_t fmt_hash, int32_t arg_count,
                 const uint8_t* etypes, uint32_t values_size,
                 const uint8_t* values)
{
    auto* sg = reinterpret_cast<ShaderGlobals*>(sg_void);
    sg->renderer->errorfmt(sg, ustringhash(fmt_hash), arg_count,
                           reinterpret_cast<const EncodedType*>(etypes),
                           values_size, const_cast<uint8_t*>(values));
}



extern "C" OSL_DLL_EXPORT void
osl_cpp_warningfmt(void* sg_void, uint64_t fmt_hash, int32_t arg_count,
                   const uint8_t* etypes, uint32_t values_size,
                   const uint8_t* values)
{
    auto* sg = reinterpret_cast<ShaderGlobals*>(sg_void);
    sg->renderer->warningfmt(sg, ustringhash(fmt_hash), arg_count,
                             reinterpret_cast<const EncodedType*>(etypes),
                             values_size, const_cast<uint8_t*>(values));
}



extern "C" OSL_DLL_EXPORT void
osl_cpp_filefmt(void* sg_void, uint64_t filename_hash, uint64_t fmt_hash,
                int32_t arg_count, const uint8_t* etypes, uint32_t values_size,
                const uint8_t* values)
{
    auto* sg = reinterpret_cast<ShaderGlobals*>(sg_void);
    sg->renderer->filefmt(sg, ustringhash(filename_hash), ustringhash(fmt_hash),
                          arg_count,
                          reinterpret_cast<const EncodedType*>(etypes),
                          values_size, const_cast<uint8_t*>(values));
}



// Exported wrapper for format() op: decode_message + ustring, callable by
// generated shader DSOs.
extern "C" OSL_DLL_EXPORT OSL::ustringhash_pod
osl_cpp_formatfmt(uint64_t fmt_hash, int32_t arg_count, const uint8_t* etypes,
                  uint32_t values_size, const uint8_t* values)
{
    std::string decoded;
    decode_message(fmt_hash, arg_count,
                   reinterpret_cast<const EncodedType*>(etypes), values,
                   decoded);
    return ustring(decoded).hash();
}



// Convert a C string to a C++ string literal (escape special chars).
std::string
BackendCpp::quoted_string(string_view s) const
{
    std::string r;
    r.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
        case '\n': r += "\\n"; break;
        case '\t': r += "\\t"; break;
        case '\r': r += "\\r"; break;
        case '"': r += "\\\""; break;
        case '\\': r += "\\\\"; break;
        default:
            if (c >= 32 && c < 127)
                r += char(c);
            else
                r += fmtformat("\\x{:02x}", c);
        }
    }
    return r;
}



std::string
BackendCpp::cpp_string_literal_rep(string_view s) const
{
    return fmtformat("OSL::ustring(\"{}\").hash()", quoted_string(s));
}



// Encoded-type byte for a symbol component, matching llvm_gen_print_fmt logic.
static uint8_t
encoded_type_for(const Symbol& sym, char fchar)
{
    using ET   = OSL::EncodedType;
    TypeDesc t = sym.typespec().simpletype();
    if (t.basetype == TypeDesc::STRING)
        return uint8_t(ET::kUstringHash);
    if (t.basetype == TypeDesc::INT)
        return (fchar == 'x' || fchar == 'X') ? uint8_t(ET::kUInt32)
                                              : uint8_t(ET::kInt32);
    return uint8_t(ET::kFloat);  // FLOAT
}



std::string
BackendCpp::printf_arg_expr(const Symbol& sym, int a, int c,
                            bool* needs_temp) const
{
    TypeDesc t     = sym.typespec().simpletype();
    bool is_array  = t.arraylen != 0;
    bool is_agg    = t.aggregate > 1;
    bool is_matrix = t.aggregate == TypeDesc::MATRIX44;
    bool is_string = (t.basetype == TypeDesc::STRING
                      || t.basetype == TypeDesc::USTRINGHASH);

    *needs_temp = false;
    if (sym.is_constant() && !is_array && !is_string) {
        *needs_temp = true;
        return (t.basetype == TypeDesc::FLOAT)
                   ? float_lit(sym.get_float(c))
                   : fmtformat("{}", sym.get_int(c));
    }

    std::string base = sym.cpp_safe_name();
    if (is_array)
        base = fmtformat("{}[{}]", base, a);
    // A deriv-carrying triple is a Dual2<Vec3>; index its value component.
    if (sym_carries_derivs(sym) && is_agg)
        base += ".val()";
    // Address aggregate components by index: Vec3/Color3 expose operator[],
    // Matrix44 is row-major operator[][].  (The old `.x/.y/.z` member form only
    // covered 3-component aggregates and ran past the end for a Matrix44.)
    if (is_matrix)
        base = fmtformat("{}[{}][{}]", base, c / 4, c % 4);
    else if (is_agg)
        base = fmtformat("{}[{}]", base, c);
    return base;
}



// C++ code generator for printf, format, fprintf, warning, error.
// Mirrors llvm_gen_print_fmt: converts printf-style format to fmtlib, then
// emits OSL::printfmt/errorfmt/warningfmt/filefmt for the void variants, and
// osl_cpp_formatfmt for the string-returning format() op.
bool
cpp_gen_printf(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    ustring opname = op.opname();

    static const ustring s_format("format"), s_fprintf("fprintf"),
        s_error("error"), s_warning("warning"), s_printf("printf");

    // format and fprintf have the format string in arg 1; others in arg 0.
    int fmtarg     = (opname == s_format || opname == s_fprintf) ? 1 : 0;
    Symbol& FmtSym = *rop.opargsym(op, fmtarg);

    if (!FmtSym.is_constant()) {
        rop.shadingcontext()->warningfmt("{} requires constant format string\n",
                                         opname);
        return false;
    }

    // Convert printf-style format string to fmtlib style.
    const char* src = FmtSym.get_string().c_str();
    std::string new_fmt;
    int arg = fmtarg + 1;

    // Per-arg info accumulated for the manual encoding block.
    std::vector<uint8_t> arg_etypes;     // EncodedType bytes
    std::vector<std::string> arg_exprs;  // C++ lvalue exprs or const literals
    std::vector<int> arg_sizes;          // byte size per encoded arg
    std::vector<bool> arg_needs_temp;    // const literal -> materialize a temp

    while (*src) {
        if (*src != '%') {
            char ch = *src++;
            new_fmt += ch;
            if (ch == '{' || ch == '}')
                new_fmt += ch;  // fmtlib escape
            continue;
        }
        if (src[1] == '%') {
            new_fmt += '%';
            src += 2;
            continue;
        }
        // Scan to format-specifier end char.
        const char* spec_start = src + 1;
        while (*src && !std::strchr("cdefgimnopqsuvxXEFGOSUX", *src))
            ++src;
        char fchar = *src++;  // consume specifier char

        if (arg >= op.nargs()) {
            rop.shadingcontext()->errorfmt("printf: format/arg mismatch ({}:{})",
                                           op.sourcefile(), op.sourceline());
            return false;
        }

        Symbol& sym = *rop.opargsym(op, arg++);
        TypeDesc td = sym.typespec().simpletype();
        int nelems  = td.numelements();
        int ncomps  = td.aggregate;

        // Build the fmtlib specifier for this slot (strip leading %).
        std::string spec(spec_start, src - 1);  // between % and fchar
        // Coerce type mismatches (same logic as llvm_gen_print_fmt).
        if (td.basetype == TypeDesc::INT && fchar != 'd' && fchar != 'i'
            && fchar != 'o' && fchar != 'u' && fchar != 'x' && fchar != 'X')
            fchar = 'd';
        if (td.basetype == TypeDesc::FLOAT && fchar != 'f' && fchar != 'g'
            && fchar != 'e')
            fchar = 'f';
        if ((td.basetype == TypeDesc::STRING) && fchar != 's')
            fchar = 's';
        // fmtlib has no 'i' integer presentation type; spell it 'd'.
        if (fchar == 'i')
            fchar = 'd';
        // fmtlib: left-justify is '<', not '-'.
        auto lpos = spec.find('-');
        if (lpos != std::string::npos) {
            spec[lpos] = '<';
            while ((lpos = spec.find('-')) != std::string::npos)
                spec.erase(lpos, 1);
        }
        std::string slot = "{:" + spec + fchar + "}";

        // A closure prints as its string form: convert to a ustringhash at
        // runtime and encode it as a single string arg (mirrors
        // llvm_gen_print_fmt). needs_temp materializes the 8-byte hash so its
        // address can be taken for the value buffer.
        if (sym.typespec().is_closure_based()) {
            new_fmt += slot;
            arg_etypes.push_back(uint8_t(OSL::EncodedType::kUstringHash));
            arg_exprs.push_back(
                fmtformat("osl_closure_to_ustringhash((void*)sg, {})",
                          rop.cpp_value_str(sym)));
            arg_needs_temp.push_back(true);
            arg_sizes.push_back(8);
            continue;
        }

        for (int a = 0; a < nelems; ++a) {
            for (int c = 0; c < ncomps; ++c) {
                if (a != 0 || c != 0)
                    new_fmt += ' ';
                new_fmt += slot;

                // Collect per-arg info.
                uint8_t et = encoded_type_for(sym, fchar);
                arg_etypes.push_back(et);
                bool needs_temp = false;
                arg_exprs.push_back(
                    rop.printf_arg_expr(sym, a, c, &needs_temp));
                arg_needs_temp.push_back(needs_temp);

                using ET = OSL::EncodedType;
                int sz   = (et == uint8_t(ET::kUstringHash)) ? 8 : 4;
                arg_sizes.push_back(sz);
            }
        }
    }

    // Prepend "Shader op [name]: " for error/warning.
    if (opname == s_error || opname == s_warning)
        new_fmt = fmtformat("Shader {} [{}]: {}", opname,
                            rop.inst()->shadername(), new_fmt);

    // All ops: emit a block that packs args into a buffer and calls the
    // appropriate osl_cpp_*fmt exported wrapper.  rs_printfmt and friends have
    // hidden visibility and cannot be bound by generated DSOs; the wrappers
    // route through the RendererServices virtual methods.
    int total_sz = 0;
    for (int sz : arg_sizes)
        total_sz += sz;
    int nargs = (int)arg_etypes.size();

    std::string osl_fn;
    if (opname == s_printf)
        osl_fn = "osl_cpp_printfmt";
    else if (opname == s_error)
        osl_fn = "osl_cpp_errorfmt";
    else if (opname == s_warning)
        osl_fn = "osl_cpp_warningfmt";
    else if (opname == s_fprintf)
        osl_fn = "osl_cpp_filefmt";
    else
        osl_fn = "osl_cpp_formatfmt";  // format

    // Determine the result symbol (format op only).
    Symbol* Result = (opname == s_format) ? rop.opargsym(op, 0) : nullptr;

    rop.outputfmtln("{{");
    rop.increment_indent();

    // Emit the format-string hash as a static const so OSL::ustring(...).hash()
    // runs once (interning the string and caching the hash) rather than on
    // every shader invocation.  The static is function-scoped so each call
    // site gets its own guard and there are no name collisions.
    rop.outputfmtln("static const uint64_t _fmthash = {};",
                    rop.cpp_string_literal_rep(new_fmt));
    std::string fmt_lit = "_fmthash";

    if (nargs > 0) {
        // EncodedType byte array.
        std::string et_list;
        for (int i = 0; i < nargs; ++i)
            et_list += fmtformat("{}{}u", i ? "," : "", arg_etypes[i]);
        rop.outputfmtln("const uint8_t _et[] = {{ {} }};", et_list);
        // Value buffer via memcpy.
        rop.outputfmtln("uint8_t _av[{}];", total_sz);
        int off = 0;
        for (int i = 0; i < nargs; ++i) {
            std::string addr = arg_exprs[i];
            if (arg_needs_temp[i]) {
                // Inlined constant literal: materialize a temp to take its
                // address.  Type follows the encoded type (4-byte int/float).
                using ET = OSL::EncodedType;
                const char* ctype
                    = (arg_etypes[i] == uint8_t(ET::kFloat))    ? "float"
                      : (arg_etypes[i] == uint8_t(ET::kUInt32)) ? "uint32_t"
                      : (arg_etypes[i] == uint8_t(ET::kUstringHash))
                          ? "OSL::ustringhash_pod"
                          : "int32_t";
                addr = fmtformat("_ac{}", i);
                rop.outputfmtln("{} {} = {};", ctype, addr, arg_exprs[i]);
            }
            rop.outputfmtln("std::memcpy(_av+{}, &{}, {});", off, addr,
                            arg_sizes[i]);
            off += arg_sizes[i];
        }
    }

    // Build call string.
    std::string call;
    if (opname == s_fprintf) {
        Symbol& Fn = *rop.opargsym(op, 0);
        // The filename is a ustringhash variable but osl_cpp_filefmt takes it as
        // a uint64 hash; pass the pod (cpp_spacename_pod adds .hash() for non-const).
        call = fmtformat("sg, {}, {}, {}, {}, {}u, {}",
                         rop.cpp_spacename_pod(Fn), fmt_lit, nargs,
                         nargs ? "_et" : "nullptr", total_sz,
                         nargs ? "_av" : "nullptr");
    } else if (opname == s_format) {
        call = fmtformat("{}, {}, {}, {}u, {}", fmt_lit, nargs,
                         nargs ? "_et" : "nullptr", total_sz,
                         nargs ? "_av" : "nullptr");
    } else {
        call = fmtformat("sg, {}, {}, {}, {}u, {}", fmt_lit, nargs,
                         nargs ? "_et" : "nullptr", total_sz,
                         nargs ? "_av" : "nullptr");
    }

    if (Result)
        // format() returns a ustringhash_pod; the Result is a ustringhash.
        rop.outputfmtln("{} = OSL::ustringhash::from_hash({}({}));",
                        Result->cpp_safe_name(), osl_fn, call);
    else
        rop.outputfmtln("{}({});", osl_fn, call);

    rop.decrement_indent();
    rop.outputfmtln("}}");
    return true;
}



// C++ code generator for sincos(theta, sin_out, cos_out).
// Two output args require void* passing; mirrors llvm_gen_sincos encoding.
bool
cpp_gen_sincos(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Theta      = *rop.opargsym(op, 0);
    Symbol& Sin_out    = *rop.opargsym(op, 1);
    Symbol& Cos_out    = *rop.opargsym(op, 2);
    bool theta_deriv   = Theta.has_derivs();
    bool result_derivs = Sin_out.has_derivs() || Cos_out.has_derivs();

    // Build function name: osl_sincos_ + per-arg (d?)type encoding
    std::string name = "osl_sincos_";
    for (int i = 0; i < op.nargs(); ++i) {
        Symbol* s = rop.opargsym(op, i);
        if (s->has_derivs() && result_derivs && theta_deriv)
            name += "d";
        if (s->typespec().is_float())
            name += "f";
        else if (s->typespec().is_triple())
            name += "v";
    }

    // Theta: by value for plain float (no derivs, not triple); else void*
    bool theta_by_ptr = (theta_deriv && result_derivs)
                        || Theta.typespec().is_triple();
    std::string theta_arg = theta_by_ptr
                                ? fmtformat("(void*)&{}", Theta.cpp_safe_name())
                                : rop.cpp_value_str(Theta);

    rop.outputfmtln("{}({}, (void*)&{}, (void*)&{});", name, theta_arg,
                    Sin_out.cpp_safe_name(), Cos_out.cpp_safe_name());
    return true;
}



// C++ code generator for break and continue.
bool
cpp_gen_loopmod_op(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    static ustring op_break("break");
    if (op.opname() == op_break) {
        rop.outputfmtln("break;");
    } else {  // continue
        const std::string& tgt = rop.loop_cont_target();
        if (tgt.empty())
            rop.outputfmtln("continue;");
        else
            rop.outputfmtln("goto {};", tgt);
    }
    return true;
}



// C++ code generator for return and exit.
//   'exit'   always leaves the whole layer function -> natural 'return;'.
//   'return' inside an inlined functioncall body jumps to the body's end label;
//            at the top level it leaves the layer function -> natural 'return;'.
bool
cpp_gen_return(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    static ustring s_exit("exit");
    // A function-scope `return` jumps to that function's return label.
    if (op.opname() != s_exit && rop.inside_function()
        && !rop.func_return_target().empty()) {
        rop.outputfmtln("goto {};", rop.func_return_target());
        return true;
    }
    // A shader-scope return (or exit) must leave via the layer exit label, not a
    // bare `return;`, so the output/groupdata/global write-backs that follow the
    // main code still run (mirrors the JIT branching to its exit_instance block
    // before the output-copy pass).
    rop.outputfmtln("goto cpp_layer_exit;");
    return true;
}



// Return the C++ index expression to use for an indexed access, wrapping it in
// osl_range_check when the shader has range checking enabled and the index is
// not a provably in-range constant (mirrors the range-check blocks in the JIT's
// llvm_gen_aref/aassign/compref/compassign/mxcomp*). osl_range_check reports an
// out-of-range error and returns a clamped index; a provably in-range constant
// (or range checking off) returns the raw index unchanged. `symname` is the name
// shown in the error message (the array/matrix symbol).
std::string
BackendCpp::cpp_range_check(const Opcode& op, const Symbol& Index, int length,
                            string_view symname)
{
    std::string idx = cpp_value_str(Index);
    if (!inst()->master()->range_checking())
        return idx;
    if (Index.is_constant() && Index.get_int() >= 0 && Index.get_int() < length)
        return idx;
    return fmtformat(
        "osl_range_check({}, {}, {}, (void*)sg, {}, {}, {}, {}, {}, {})", idx,
        length, cpp_string_literal_rep(symname),
        cpp_string_literal_rep(op.sourcefile()), op.sourceline(),
        cpp_string_literal_rep(group().name()), layer(),
        cpp_string_literal_rep(inst()->layername()),
        cpp_string_literal_rep(inst()->shadername()));
}



// Array element reference: Result = Src[Index]. Native C++ operator[] on the
// generated array type handles both constant and runtime indices uniformly.
bool
cpp_gen_aref(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));
    Symbol& Index(*rop.opargsym(op, 2));
    std::string idx = rop.cpp_range_check(op, Index,
                                          Src.typespec().arraylength(),
                                          Src.unmangled());
    // A deriv-carrying array element is a Dual2; if the result drops derivs,
    // read just the value (the reverse, Dual2 from a plain element, is an
    // implicit widening that zeroes derivs).
    std::string elem = fmtformat("{}[{}]", rop.cpp_value_str(Src), idx);
    if (rop.sym_carries_derivs(Src) && !rop.sym_carries_derivs(R))
        elem += ".val()";
    rop.outputfmtln("{} = {};", R.cpp_safe_name(), elem);
    return true;
}



// Array element assignment: Result[Index] = Val.
bool
cpp_gen_aassign(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& Index(*rop.opargsym(op, 1));
    Symbol& Val(*rop.opargsym(op, 2));
    // String element from a string constant: the element is ustringhash but the
    // constant is a raw uint64_t hash, so wrap it (mirrors cpp_gen_assign).
    std::string val = rop.cpp_value_str(Val);
    if (R.typespec().simpletype().basetype == TypeDesc::STRING
        && Val.symtype() == SymTypeConst)
        val = fmtformat("OSL::ustringhash::from_hash({})", val);
    std::string idx = rop.cpp_range_check(op, Index, R.typespec().arraylength(),
                                          R.unmangled());
    rop.outputfmtln("{}[{}] = {};", R.cpp_safe_name(), idx, val);
    return true;
}



// Vector/color component reference: Result = Val[Index]. Kept separate from
// aref so a future revision can emit `.x`/`.y`/`.z` for constant indices.
bool
cpp_gen_compref(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& Val(*rop.opargsym(op, 1));
    Symbol& Index(*rop.opargsym(op, 2));
    // Range-check the index (length 3). When a check is inserted, materialize it
    // into a temp so the index is evaluated once (one error report) even though
    // the Dual2 path uses it three times.
    std::string i = rop.cpp_range_check(op, Index, 3, Val.unmangled());
    if (i != rop.cpp_value_str(Index)) {
        rop.outputfmtln("int ___cidx{} = {};", opnum, i);
        i = fmtformat("___cidx{}", opnum);
    }
    // A deriv-carrying triple is a Dual2<Vec3> with no operator[]; index its
    // value (and, when the result keeps derivatives, the partials too).
    if (rop.sym_carries_derivs(Val) && Val.typespec().is_triple()) {
        std::string b = Val.cpp_safe_name();
        if (rop.sym_carries_derivs(R))
            rop.outputfmtln(
                "{} = OSL::Dual2<float>({}.val()[{}], {}.dx()[{}], {}.dy()[{}]);",
                R.cpp_safe_name(), b, i, b, i, b, i);
        else
            rop.outputfmtln("{} = {}.val()[{}];", R.cpp_safe_name(), b, i);
        return true;
    }
    rop.outputfmtln("{} = {}[{}];", R.cpp_safe_name(), rop.cpp_value_str(Val),
                    i);
    return true;
}



// Vector/color component assignment: Result[Index] = Val.
bool
cpp_gen_compassign(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& Index(*rop.opargsym(op, 1));
    Symbol& Val(*rop.opargsym(op, 2));
    // Range-check the index (length 3); materialize when a check is inserted so
    // it is evaluated once even though the Dual2 path uses it three times.
    std::string i = rop.cpp_range_check(op, Index, 3, R.unmangled());
    if (i != rop.cpp_value_str(Index)) {
        rop.outputfmtln("int ___cidx{} = {};", opnum, i);
        i = fmtformat("___cidx{}", opnum);
    }
    // Assigning one component of a deriv-carrying triple (Dual2<Vec3>): update
    // that component of the value and of each stored partial.
    if (rop.sym_carries_derivs(R) && R.typespec().is_triple()) {
        std::string b = R.cpp_safe_name();
        if (rop.sym_carries_derivs(Val)) {
            std::string v = rop.cpp_value_str(Val);
            rop.outputfmtln("{}.val()[{}] = {}.val();", b, i, v);
            rop.outputfmtln("{}.dx()[{}] = {}.dx();", b, i, v);
            rop.outputfmtln("{}.dy()[{}] = {}.dy();", b, i, v);
        } else {
            rop.outputfmtln("{}.val()[{}] = {};", b, i, rop.cpp_float_val(Val));
            rop.outputfmtln("{}.dx()[{}] = 0.0f;", b, i);
            rop.outputfmtln("{}.dy()[{}] = 0.0f;", b, i);
        }
        return true;
    }
    rop.outputfmtln("{}[{}] = {};", R.cpp_safe_name(), i,
                    rop.cpp_value_str(Val));
    return true;
}



// Matrix component reference: Result = M[Row][Col].
bool
cpp_gen_mxcompref(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& M(*rop.opargsym(op, 1));
    Symbol& Row(*rop.opargsym(op, 2));
    Symbol& Col(*rop.opargsym(op, 3));
    // Each index used once, so an inline check (length 4) is fine.
    std::string row = rop.cpp_range_check(op, Row, 4, M.name());
    std::string col = rop.cpp_range_check(op, Col, 4, M.name());
    rop.outputfmtln("{} = {}[{}][{}];", R.cpp_safe_name(), rop.cpp_value_str(M),
                    row, col);
    return true;
}



// Matrix component assignment: Result[Row][Col] = Val.
bool
cpp_gen_mxcompassign(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& Row(*rop.opargsym(op, 1));
    Symbol& Col(*rop.opargsym(op, 2));
    Symbol& Val(*rop.opargsym(op, 3));
    std::string row = rop.cpp_range_check(op, Row, 4, R.name());
    std::string col = rop.cpp_range_check(op, Col, 4, R.name());
    rop.outputfmtln("{}[{}][{}] = {};", R.cpp_safe_name(), row, col,
                    rop.cpp_value_str(Val));
    return true;
}



// Array length: Result = constant length of the array argument.
bool
cpp_gen_arraylength(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& A(*rop.opargsym(op, 1));
    int len = A.typespec().is_unsized_array() ? A.initializers()
                                              : A.typespec().arraylength();
    rop.outputfmtln("{} = {};", R.cpp_safe_name(), len);
    return true;
}



// Array copy: Result = Src for whole same-typed arrays. C++ arrays aren't
// assignable with '=', so emit a memcpy of the full array size.
bool
cpp_gen_arraycopy(BackendCpp& rop, int opnum)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& R(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));
    rop.cpp_array_copy(R, Src);
    return true;
}



void
BackendCpp::op_gen_init()
{
    static std::mutex mutex;  // only one BackendCpp can do this at a time
    std::lock_guard<std::mutex> lock(mutex);

    // clang-format off
#define OP(name,cg)                                                     \
    if (auto* op = shadingsys().op_descriptormod(ustring(#name))) {     \
        if (op->cppgen) return; /* already set */                       \
        op->cppgen = cpp_gen_##cg;                                      \
    }

    // print("running BackendCpp::op_gen_init()\n");

    // name          cg gen               folder         simple     flags
    OP (aassign,     aassign);
    OP (abs,         generic);
    OP (acos,        generic);
    OP (add,         binary_op);
    OP (and,         binary_op);
    OP (area,        area);
    OP (aref,        aref);
    OP (arraycopy,   arraycopy);
    OP (arraylength, arraylength);
    OP (asin,        generic);
    OP (assign,      assign);
    OP (atan,        generic);
    OP (atan2,       generic);
    OP (backfacing,  get_simple_SG_field);
    OP (bitand,      binary_op);
    OP (bitor,       binary_op);
    OP (blackbody,   blackbody);
    OP (break,       loopmod_op);
    OP (calculatenormal, calculatenormal);
    OP (cbrt,        generic);
    OP (ceil,        generic);
    OP (cellnoise,   generic /*noise*/);
    OP (clamp,       generic);
    OP (closure,     closure);
    OP (color,       construct_color);
    OP (compassign,  compassign);
    OP (compl,       unary_op);
    OP (compref,     compref);
    OP (concat,      generic);
    OP (continue,    loopmod_op);
    OP (cos,         generic);
    OP (cosh,        generic);
    OP (cross,       generic);
    OP (degrees,     generic);
    OP (determinant, generic);
    OP (dict_find,   dict_find);
    OP (dict_next,   dict_next);
    OP (dict_value,  dict_value);
    OP (distance,    generic);
    OP (div,         div);
    OP (dot,         generic);
    OP (Dx,          DxDy);
    OP (Dy,          DxDy);
    OP (Dz,          DxDy);
    OP (dowhile,     loop_op);
    OP (end,         nop);
    OP (endswith,    generic);
    OP (environment, environment);
    OP (eq,          compare_op);
    OP (erf,         generic);
    OP (erfc,        generic);
    OP (error,       printf);
    OP (exit,        return);
    OP (exp,         generic);
    OP (exp2,        generic);
    OP (expm1,       generic);
    OP (fabs,        generic);
    OP (filterwidth, filterwidth);
    OP (floor,       generic);
    OP (fmod,        generic);
    OP (for,         loop_op);
    OP (format,      printf);
    OP (fprintf,     printf);
    OP (functioncall,    functioncall);
    OP (functioncall_nr, functioncall);
    OP (ge,          compare_op);
    OP (getattribute, getattribute);
    OP (getchar,      generic);
    OP (getmatrix,   getmatrix);
    OP (getmessage,  getmessage);
    OP (gettextureinfo, gettextureinfo);
    OP (gt,          compare_op);
    OP (hash,        generic);
    OP (hashnoise,   generic /*noise*/);
    OP (if,          if);
    OP (inversesqrt, generic);
    OP (isconnected, generic);
    OP (isconstant,  isconstant);
    OP (isfinite,    generic);
    OP (isinf,       generic);
    OP (isnan,       generic);
    OP (le,          compare_op);
    OP (length,      generic);
    OP (log,         generic);
    OP (log10,       generic);
    OP (log2,        generic);
    OP (logb,        generic);
    OP (lt,          compare_op);
    OP (luminance,   luminance);
    OP (matrix,      matrix);
    OP (max,         generic);
    OP (mxcompassign, mxcompassign);
    OP (mxcompref,   mxcompref);
    OP (min,         generic);
    OP (mix,         generic);
    OP (mod,         generic);
    OP (mul,         binary_op);
    OP (neg,         unary_op);
    OP (neq,         compare_op);
    OP (noise,       noise);
    OP (nop,         nop);
    OP (normal,      construct_triple);
    OP (normalize,   generic);
    OP (or,          binary_op);
    OP (pnoise,      noise);
    OP (point,       construct_triple);
    OP (pointcloud_search, pointcloud_search);
    OP (pointcloud_get, pointcloud_get);
    OP (pointcloud_write, pointcloud_write);
    OP (pow,         generic);
    OP (printf,      printf);
    OP (psnoise,     noise);
    OP (radians,     generic);
    OP (raytype,     raytype);
    OP (regex_match, regex);
    OP (regex_search, regex);
    OP (return,      return);
    OP (round,       generic);
    OP (select,      select);
    OP (setmessage,  setmessage);
    OP (shl,         binary_op);
    OP (shr,         binary_op);
    OP (sign,        generic);
    OP (sin,         generic);
    OP (sincos,      sincos);
    OP (sinh,        generic);
    OP (smoothstep,  generic);
    OP (snoise,      noise);
    OP (spline,      spline);
    OP (splineinverse, spline);
    OP (split,       split);
    OP (sqrt,        generic);
    OP (startswith,  generic);
    OP (step,        generic);
    OP (stof,        generic);
    OP (stoi,        generic);
    OP (strlen,      generic);
    OP (strtof,      generic);
    OP (strtoi,      generic);
    OP (sub,         binary_op);
    OP (substr,      generic);
    OP (surfacearea, get_simple_SG_field);
    OP (tan,         generic);
    OP (tanh,        generic);
    OP (texture,     texture);
    OP (texture3d,   texture3d);
    OP (trace,       trace);
    OP (transform,   transform);
    OP (transformc,  transformc);
    OP (transformn,  transform);
    OP (transformv,  transform);
    OP (transpose,   generic);
    OP (trunc,       generic);
    OP (useparam,    useparam);
    OP (vector,      construct_triple);
    OP (warning,     printf);
    OP (wavelength_color, blackbody);
    OP (while,       loop_op);
    OP (xor,         binary_op);
#undef OP
#undef OP2
    // clang-format on
}


};  // namespace pvt
OSL_NAMESPACE_END
