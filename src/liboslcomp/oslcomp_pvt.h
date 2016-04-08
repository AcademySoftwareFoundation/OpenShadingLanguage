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
#include <stack>
#include <set>
#include <map>

#include "OSL/oslcomp.h"
#include "ast.h"
#include "symtab.h"
#include "OSL/genclosure.h"


extern int oslparse ();


OSL_NAMESPACE_ENTER

namespace pvt {



/// Set of symbols, identified by pointers.
///
typedef std::set<const Symbol *> SymPtrSet;

/// For each symbol, have a list of the symbols it depends on (or that
/// depends on it).
typedef std::map<const Symbol *, SymPtrSet> SymDependencyMap;



class OSLCompilerImpl {
public:
    OSLCompilerImpl (ErrorHandler *errhandler);
    ~OSLCompilerImpl ();

    /// Fully compile a shader located in 'filename', with the command-line
    /// options ("-I" and the like) in the options vector.
    bool compile (string_view filename,
                  const std::vector<std::string> &options,
                  string_view stdoslpath);

    bool compile_buffer (string_view sourcecode,
                         std::string &osobuffer,
                         const std::vector<std::string> &options,
                         string_view stdoslpath);

    bool osl_parse_buffer (const std::string &preprocessed_buffer);

    /// The name of the file we're currently parsing
    ///
    ustring filename () const { return m_filename; }

    /// Set the name of the file we're currently parsing (should only
    /// be called by the lexer!)
    void filename (ustring f) { m_filename = f; }

    /// The line we're currently parsing
    ///
    int lineno () const { return m_lineno; }

    /// Set the line we're currently parsing (should only be called by
    /// the lexer!)
    void lineno (int l) { m_lineno = l; }

    /// Increment the line count
    ///
    int incr_lineno () { return ++m_lineno; }

    ErrorHandler &errhandler () const { return *m_errhandler; }

    /// Error reporting
    ///
    void error (ustring filename, int line, const char *format, ...) const;

    /// Warning reporting
    ///
    void warning (ustring filename, int line, const char *format, ...) const;

    /// Have we hit an error?
    ///
    bool error_encountered () const { return m_err; }

    /// Look at the compile options, setting defines, includepaths, and
    /// a variety of other private options.
    void read_compile_options (const std::vector<std::string> &options,
                               std::vector<std::string> &defines,
                               std::vector<std::string> &includepaths);

    bool preprocess_file (const std::string &filename,
                          const std::string &stdoslpath,
                          const std::vector<std::string> &defines,
                          const std::vector<std::string> &includepaths,
                          std::string &result);

    bool preprocess_buffer (const std::string &buffer,
                            const std::string &filename,
                            const std::string &stdoslpath,
                            const std::vector<std::string> &defines,
                            const std::vector<std::string> &includepaths,
                            std::string &result);

    /// Has a shader already been defined?
    bool shader_is_defined () const { return (bool)m_shader; }

    /// Define the shader we're compiling with the given AST root.
    ///
    void shader (ASTNode::ref s) { m_shader = s; }

    /// Return the AST root of the main shader we're compiling.
    ///
    ASTNode::ref shader () const { return m_shader; }

    /// Return a reference to the symbol table.
    ///
    SymbolTable &symtab () { return m_symtab; }
    const SymbolTable &symtab () const { return m_symtab; }

    TypeSpec current_typespec () const { return m_current_typespec; }
    void current_typespec (TypeSpec t) { m_current_typespec = t; }
    bool current_output () const { return m_current_output; }
    void current_output (bool b) { m_current_output = b; }

    void declaring_shader_formals (bool val) { m_declaring_shader_formals = val; }
    bool declaring_shader_formals () const { return m_declaring_shader_formals; }

    /// Given a pointer to a type code string that we use for argument
    /// checking ("p", "v", etc.) return the TypeSpec of the first type
    /// described by the string (UNKNOWN if it couldn't be recognized).
    /// If 'advance' is non-NULL, set *advance to the number of
    /// characters taken by the first code so the caller can advance
    /// their pointer to the next code in the string.
    static TypeSpec type_from_code (const char *code, int *advance=NULL);

    /// Return the argument checking code ("p", "v", etc.) corresponding
    /// to the type.
    std::string code_from_type (TypeSpec type) const;

    /// Take a type code string (possibly containing many types)
    /// and turn it into a human-readable string.
    std::string typelist_from_code (const char *code) const;

    /// Take a type code string (possibly containing many types) and
    /// turn it into a TypeSpec vector.
    void typespecs_from_codes (const char *code,
                               std::vector<TypeSpec> &types) const;

    /// Emit a single IR opcode -- append one op to the list of
    /// intermediate code, returning the label (address) of the new op.
    int emitcode (const char *opname, size_t nargs, Symbol **args,
                  ASTNode *node);

    /// Insert a new opcode in front of the desired position.  Then
    /// it's necessary to adjust all the jump targets!
    int insert_code (int position, const char *opname,
                     size_t nargs, Symbol **args, ASTNode *node);

    /// Return the label (opcode address) for the next opcode that will
    /// be emitted.
    int next_op_label () { return (int)m_ircode.size(); }

    /// Add op arguments, return the index of the first one added.
    /// Use with extreme caution!  If you're not the guts of 'emitcode',
    /// think twice about how you use this so you don't "leak" arguments.
    size_t add_op_args (size_t nargs, Symbol **args);

    /// Return a reference to the last opcode that we added.
    ///
    Opcode & lastop () { return m_ircode.back(); }

    /// Return a reference to a given IR opcode.
    ///
    Opcode & ircode (int index) { return m_ircode[index]; }

    /// Return a reference to the full opargs list.
    ///
    SymbolPtrVec & opargs () { return m_opargs; }

    /// Specify that subsequent opcodes are for a particular method
    ///
    void codegen_method (ustring method);

    /// Which method or parameter is currently undergoing code generation?
    /// "___main___" indicates the main body of code.
    ustring codegen_method () const { return m_codegenmethod; }

    /// Return the name of the 'main' method.
    ///
    static ustring main_method_name ();

    /// Make a temporary symbol of the given type.
    ///
    Symbol *make_temporary (const TypeSpec &type);

    /// Make a generic constant symbol
    ///
    Symbol *make_constant (TypeDesc type, const void *val);

    /// Make a constant string symbol
    ///
    Symbol *make_constant (ustring s);

    /// Make a constant int symbol
    ///
    Symbol *make_constant (int i);

    /// Make a constant float symbol
    ///
    Symbol *make_constant (float f);

    /// Make a constant triple symbol
    ///
    Symbol *make_constant (TypeDesc type, float x, float y, float z);

    // Make and add individual symbols for each field of a structure,
    // using the given basename.
    void add_struct_fields (StructSpec *structspec, ustring basename,
                            SymType symtype, int arraylen, ASTNode *node=NULL);

    string_view output_filename () const { return m_output_filename; }

    /// Push the designated function on the stack, to keep track of
    /// nesting and so recursed methods can query which is the current
    /// function in play.
    void push_function (FunctionSymbol *func) {
        m_function_stack.push (func);
        func->init_nesting ();
    }

    /// Restore the function stack to its state before the last
    /// push_function().
    void pop_function () { m_function_stack.pop (); }

    /// Return the symbol of the current user function we're descending
    /// into, or NULL if we are not inside a user function.
    FunctionSymbol *current_function () const {
        return m_function_stack.empty () ? NULL : m_function_stack.top ();
    }

    /// Push the conditional nesting level, called any time the compiler
    /// is about to recursively descend into processing (type checking,
    /// code generation, whatever) for something that may affect which
    /// points are run.  Pass true for 'isloop' if it's a loop.
    void push_nesting (bool isloop=false);

    /// Restore the previous conditional nesting level, called any time
    /// the compiler is done recursively descending.  Pass true for
    /// 'isloop' if it's a loop.
    void pop_nesting (bool isloop=false);

    /// Return the current nesting level (JUST for loops, if loops=true).
    ///
    int nesting_level (bool loops=false) const {
        return loops ? m_loop_nesting : m_total_nesting;
    }

    /// Return the c_str giving a human-readable name of a type, fully
    /// accounting for exotic types like structs, etc.
    const char *type_c_str (const TypeSpec &type) const;

    /// Given symbols sym1 and sym2, both the same kind of struct, and the
    /// index of a field we're interested, find the symbols that represent
    /// that field in the each sym and place them in field1 and field2,
    /// respectively.
    void struct_field_pair (Symbol *sym1, Symbol *sym2, int fieldnum,
                            Symbol * &field1, Symbol * &field2);

    /// Given symbol names sym1 and sym2, both the same kind of struct
    /// described by structspec, and the index of the structure field
    /// we're interested in, find the symbols that represent that field
    /// in the each sym[12] and place them in field1 and field2,
    /// respectively.
    void struct_field_pair (const StructSpec *structspec, int fieldnum,
                            ustring sym1, ustring sym2,
                            Symbol * &field1, Symbol * &field2);

    static void track_variable_lifetimes (const OpcodeVec &ircode,
                                          const SymbolPtrVec &opargs,
                                          const SymbolPtrVec &allsyms,
                                          std::vector<int> *bblock_ids=NULL);
    static void coalesce_temporaries (SymbolPtrVec &symtab);

    const std::string main_filename () const { return m_main_filename; }
    const std::string cwd () const { return m_cwd; }

private:
    void initialize_globals ();
    void initialize_builtin_funcs ();
    std::string default_output_filename ();
    void write_oso_file (const std::string &outfilename, string_view options);
    void write_oso_const_value (const ConstantSymbol *sym) const;
    void write_oso_symbol (const Symbol *sym);
    void write_oso_metadata (const ASTNode *metanode) const;
    // void oso (const char *fmt, ...) const;
    TINYFORMAT_WRAP_FORMAT (void, oso, const, , (*m_osofile), )

    void track_variable_lifetimes () {
        track_variable_lifetimes (m_ircode, m_opargs, symtab().allsyms());
    }

    void track_variable_dependencies ();
    void coalesce_temporaries () {
        coalesce_temporaries (m_symtab.allsyms());
    }

    /// Scan through all the ops and make sure none of them write to
    /// things that are illegal (consts, non-output params, etc.).
    /// Must be called AFTER track_variable_lifetimes.
    void check_for_illegal_writes ();

    /// Helper for check_for_illegal_writes: check one statement and one
    /// symbol.
    void check_write_legality (const Opcode &op, int opnum, const Symbol *sym);

    /// Does this read or write the symbol identified by 'sym'?  The
    /// optional 'read' and 'write' arguments determine whether it is
    /// considering reading or writing (or, by default, both).
    bool op_uses_sym (const Opcode &op, const Symbol *sym,
                      bool read=true, bool write=true);
    /// Does this arg read the symbol identified by 'sym'?
    ///
    bool op_reads_sym (const Opcode &op, const Symbol *sym) {
        return op_uses_sym (op, sym, true, false);
    }
    /// Does this arg read the symbol identified by 'sym'?
    ///
    bool op_writes_sym (const Opcode &op, const Symbol *sym) {
        return op_uses_sym (op, sym, false, true);
    }

    /// Add all symbols used in the op range [opbegin,opend) to rsyms
    /// and/or wsyms.  Pass NULL if you don't care abou one or the other.
    /// Pass both pointers to the same vector if you want both reads and
    /// writes recorded to the same place.
    void syms_used_in_op_range (OpcodeVec::const_iterator opbegin,
                                OpcodeVec::const_iterator opend,
                                std::vector<Symbol *> *rsyms,
                                std::vector<Symbol *> *wsyms);

    ASTshader_declaration *shader_decl () const {
        if (m_shader.get()->nodetype() != ASTNode::shader_declaration_node)
            return NULL;
        return static_cast<ASTshader_declaration *>(m_shader.get());
    }
    std::string retrieve_source (ustring filename, int line);

    ustring m_filename;       ///< Current file we're parsing
    int m_lineno;             ///< Current line we're parsing
    std::string m_output_filename; ///< Output filename
    std::string m_main_filename; ///< Main input filename
    std::string m_cwd;        ///< Current working directory
    ASTNode::ref m_shader;    ///< The shader's syntax tree
    ErrorHandler *m_errhandler; ///< Error handler
    mutable bool m_err;       ///< Has an error occurred?
    SymbolTable m_symtab;     ///< Symbol table
    TypeSpec m_current_typespec;  ///< Currently-declared type
    bool m_current_output;        ///< Currently-declared output status
    bool m_verbose;           ///< Verbose mode
    bool m_quiet;             ///< Quiet mode
    bool m_debug;             ///< Debug mode
    bool m_preprocess_only;   ///< Preprocess only?
    int m_optimizelevel;      ///< Optimization level
    OpcodeVec m_ircode;       ///< Generated IR code
    SymbolPtrVec m_opargs;    ///< Arguments for all instructions
    int m_next_temp;          ///< Next temporary symbol index
    int m_next_const;         ///< Next const symbol index
    std::vector<ConstantSymbol *> m_const_syms;  ///< All consts we've made
    std::ostream *m_osofile;  ///< Open .oso stream for output
    FILE *m_sourcefile;       ///< Open file handle for retrieve_source
    ustring m_last_sourcefile;///< Last filename for retrieve_source
    int m_last_sourceline;    ///< Last line read for retrieve_source
    ustring m_codegenmethod;  ///< Current method we're generating code for
    std::stack<FunctionSymbol *> m_function_stack; ///< Stack of called funcs
    int m_total_nesting;      ///< total conditional nesting level (0 == none)
    int m_loop_nesting;       ///< just loop nesting level (0 == none)
    SymDependencyMap m_symdeps; ///< Symbol-to-symbol dependencies
    Symbol *m_derivsym;       ///< Pseudo-symbol to track deriv dependencies
    int m_main_method_start;  ///< Instruction where 'main' starts
    bool m_declaring_shader_formals; ///< Are we declaring shader formals?
};


extern OSLCompilerImpl *oslcompiler;


}; // namespace pvt

OSL_NAMESPACE_EXIT
