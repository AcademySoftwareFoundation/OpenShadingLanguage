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


class OSLCompilerImpl : public OSL::OSLCompiler {
public:
    OSLCompilerImpl (void) : m_lexer(NULL), m_err(false) { }
    virtual ~OSLCompilerImpl (void) { }

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
    int incr_lineno () { ++m_lineno; }

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

private:
    oslFlexLexer *m_lexer;    ///< Lexical scanner
    ustring m_filename;       ///< Current file we're parsing
    int m_lineno;             ///< Current line we're parsing
    ASTNode::ref m_shader;    ///< The shader's syntax tree
    bool m_err;               ///< Has an error occurred?
    SymbolTable m_symtab;     ///< Symbol table
};


extern OSLCompilerImpl *oslcompiler;


}; // namespace pvt
}; // namespace OSL


#endif /* OSLCOMP_PVT_H */
