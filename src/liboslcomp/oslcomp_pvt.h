/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/

#ifndef OSLCOMP_PVT_H
#define OSLCOMP_PVT_H

#include "OpenImageIO/ustring.h"

#include "oslcomp.h"
#include "ast.h"
#include "symtab.h"


class oslFlexLexer;
extern int oslparse ();


namespace OSL {
namespace pvt {


class ASTNode;


/// Intermediate Represenatation opcode
///
class IROpcode {
public:
    IROpcode (ustring op, ASTNode *node) : m_op(op), m_astnode(node) { }
    void add_arg (Symbol *arg) { m_args.push_back (arg->dealias()); }
    size_t nargs () const { return m_args.size(); }
    Symbol *arg (int i) const { return m_args[i]; }
    ASTNode *node () const { return m_astnode; }
    const char *opname () const { return m_op.c_str(); }

private:
    ustring m_op;                   ///< Name of opcode
    std::vector<Symbol *> m_args;   ///< Arguments
    ASTNode *m_astnode;             ///< AST node that generated this op
};


typedef std::vector<IROpcode> IROpcodeVec;



class OSLCompilerImpl : public OSL::OSLCompiler {
public:
    OSLCompilerImpl ();
    virtual ~OSLCompilerImpl ();

    /// Fully compile a shader located in 'filename', with the command-line
    /// options ("-I" and the like) in the options vector.
    virtual bool compile (const std::string &filename,
                          const std::vector<std::string> &options);

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

    /// Return a pointer to the current lexer.
    ///
    oslFlexLexer *lexer() const { return m_lexer; }

    /// Error reporting
    ///
    void error (ustring filename, int line, const char *format, ...);

    /// Warning reporting
    ///
    void warning (ustring filename, int line, const char *format, ...);

    /// Have we hit an error?
    ///
    bool error_encountered () const { return m_err; }

    /// Has a shader already been defined?
    bool shader_is_defined () const { return m_shader; }

    /// Define the shader we're compiling with the given AST root.
    ///
    void shader (ASTNode::ref s) { m_shader = s; }

    /// Return the AST root of the main shader we're compiling.
    ///
    ASTNode::ref shader () const { return m_shader; }

    /// Return a reference to the symbol table.
    ///
    SymbolTable &symtab () { return m_symtab; }

    /// Register a symbol
    ///
//    void add_function (Symbol *sym) { m_allfuncs.push_back (sym); }

    TypeSpec current_typespec () const { return m_current_typespec; }
    void current_typespec (TypeSpec t) { m_current_typespec = t; }
    bool current_output () const { return m_current_output; }
    void current_output (bool b) { m_current_output = b; }

    /// Given a pointer to a type code string that we use for argument
    /// checking ("p", "v", etc.) return the TypeSpec of the first type
    /// described by the string (UNKNOWN if it couldn't be recognized).
    /// If 'advance' is non-NULL, set *advance to the number of
    /// characters taken by the first code so the caller can advance
    /// their pointer to the next code in the string.
    TypeSpec type_from_code (const char *code, int *advance=NULL);

    /// Take a type code string (possibly containing many types)
    /// and turn it into a human-readable string.
    std::string typelist_from_code (const char *code);

    /// Emit a single IR opcode.
    ///
    void emitcode (const char *opname, size_t nargs, Symbol **args,
                   ASTNode *node);

    Symbol *make_temporary (const TypeSpec &type);

    std::string output_filename (const std::string &inputfilename);

private:
    void initialize_globals ();
    void initialize_builtin_funcs ();
    void write_oso_file (const std::string &outfilename);
    void oso (const char *fmt, ...);
    ASTshader_declaration *shader_decl () const {
        return dynamic_cast<ASTshader_declaration *>(m_shader.get());
    }
    std::string retrieve_source (ustring filename, int line);

    oslFlexLexer *m_lexer;    ///< Lexical scanner
    ustring m_filename;       ///< Current file we're parsing
    int m_lineno;             ///< Current line we're parsing
    ASTNode::ref m_shader;    ///< The shader's syntax tree
    bool m_err;               ///< Has an error occurred?
    SymbolTable m_symtab;     ///< Symbol table
    TypeSpec m_current_typespec;  ///< Currently-declared type
    bool m_current_output;        ///< Currently-declared output status
//    SymbolList m_allfuncs;      ///< All function symbols, in decl order
    bool m_verbose;           ///< Verbose mode
    bool m_debug;             ///< Debug mode
    IROpcodeVec m_ircode;     ///< Generated IR code
    int m_next_temp;          ///< Next temporary symbol index
    FILE *m_osofile;          ///< Open .oso file for output
    FILE *m_sourcefile;       ///< Open file handle for retrieve_source
    ustring m_last_sourcefile;///< Last filename for retrieve_source
    int m_last_sourceline;    ///< Last line read for retrieve_source
};


extern OSLCompilerImpl *oslcompiler;


}; // namespace pvt
}; // namespace OSL


#endif /* OSLCOMP_PVT_H */
