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

#include "osl_pvt.h"

#include <OpenImageIO/thread.h>



class osoFlexLexer;
extern int osoparse ();


OSL_NAMESPACE_ENTER

namespace pvt {


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

    /// Return true for parsers whose only purpose is to read the header up
    /// to params, to stop parsing as soon as we start encountering temps in
    /// the symbol table.
    virtual bool stop_parsing_at_temp_symbols () { return false; }

    /// Add a hint.
    ///
    virtual void hint (const char *hintstring) { }

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

    /// Pointer to the one and only lexer in effect.  This is 'public',
    /// but NOBODY should modify this except for this class and the
    /// lexer internals.
    static osoFlexLexer *osolexer;

    static OSOReader *osoreader;

private:
    ErrorHandler &m_err;
    int m_lineno;
    static OIIO::mutex m_osoread_mutex;
};



}; // namespace pvt
OSL_NAMESPACE_EXIT
