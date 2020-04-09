// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <OSL/platform.h>
#include "osl_pvt.h"

#include <OpenImageIO/thread.h>
#include <OpenImageIO/string_view.h>


extern int osoparse ();


OSL_NAMESPACE_ENTER

namespace pvt {

// Turn off warnings about unused params, since we have lots of declarations
// with stub function bodies.
OSL_PRAGMA_WARNING_PUSH
OSL_GCC_PRAGMA(GCC diagnostic ignored "-Wunused-parameter")


/// Base class for OSO (OpenShadingLanguage object code) file reader.
///
class OSOReader {
public:
    OSOReader (ErrorHandler *errhandler = NULL) 
        : m_err (errhandler ? *errhandler : ErrorHandler::default_handler()),
          m_lineno(1)
    { }
    virtual ~OSOReader () { }

    /// Read in the oso file, parse it, call the various callbacks.
    /// Return true if the file was correctly parsed, false if there was
    /// an unrecoverable error reading the file.
    virtual bool parse_file (const std::string &filename);

    /// Read in OSO from memory, parse, call the various callbacks.
    /// Return true if the OSO code was correctly parsed, false if there was
    /// an unrecoverable error reading.
    virtual bool parse_memory (const std::string &buffer);

    /// Declare the shader version.
    ///
    virtual void version (const char *specid, int major, int minor) { }

    /// Set the name and type of the shader
    ///
    virtual void shader (const char *shadertype, const char *name) { }

    /// Register a new symbol.
    ///
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name)
    { }

    /// Add a default integer value to the last declared symbol.
    ///
    virtual void symdefault (int def) { }

    /// Add a default float value to the last declared symbol.
    ///
    virtual void symdefault (float def) { }

    /// Add a default string value to the last declared symbol.
    ///
    virtual void symdefault (const char *def) { }

    /// Called when we're done with all information related to a parameter
    /// symbol.
    virtual void parameter_done () { }

    /// Return true for parsers whose only purpose is to read the header up
    /// to params, to stop parsing as soon as we start encountering temps in
    /// the symbol table.
    virtual bool stop_parsing_at_temp_symbols () { return false; }

    /// Add a hint.
    ///
    virtual void hint (string_view hintstring) { }

    /// Return true if this parser cares about the code, false if parsing
    /// of oso may terminate once the symbol table has been parsed.
    virtual bool parse_code_section () { return true; }

    /// New code section marker designating subsequent instructions.
    ///
    virtual void codemarker (const char *name) { }

    /// Mark the end of the code section
    ///
    virtual void codeend () { }

    /// Add an instruction.
    ///
    virtual void instruction (int label, const char *opcode) { }

    /// Add an argument to the last instruction.
    ///
    virtual void instruction_arg (const char *name) { }

    /// Add a jump target to the last instruction.
    ///
    virtual void instruction_jump (int target) { }

    /// Called after an instruction (after args and hints)
    ///
    virtual void instruction_end () { }

    /// Increment the line number (for error reporting).  Should only
    /// be called by the lexer.
    void incr_lineno () { ++m_lineno; }

    /// Return the line number (for error reporting).  Should only
    /// be called by the lexer.
    int lineno () const { return m_lineno; }

    /// Return a reference to the error handler
    ErrorHandler& errhandler () { return m_err; }

    static OSOReader *osoreader;

private:
    ErrorHandler &m_err;
    int m_lineno;
};

OSL_PRAGMA_WARNING_POP


}; // namespace pvt
OSL_NAMESPACE_EXIT
