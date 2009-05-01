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

#ifndef OSOREADER_H
#define OSOREADER_H

#include "OpenImageIO/ustring.h"
#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/thread.h"

#include "osl_pvt.h"


class osoFlexLexer;
extern int osoparse ();


namespace OSL {
namespace pvt {


/// Base class for OSO (OpenShadingLanguage object code) file reader.
///
class OSOReader {
public:
    OSOReader () { }
    virtual ~OSOReader () { }

    /// Read in the oso file, parse it, call the various callbacks.
    /// Return true if the file was correctly parsed, false if there was
    /// an unrecoverable error reading the file.
    bool parse (const std::string &filename);

    /// Declare the shader version.
    ///
    virtual void version (const char *specid, float version) { }

    /// Set the name and type of the shader
    ///
    virtual void shader (const char *shadertype, const char *name) { }

    /// Register a new symbol.
    ///
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name);

    /// Add a default integer value to the last declared symbol.
    ///
    virtual void symdefault (int def) { }

    /// Add a default float value to the last declared symbol.
    ///
    virtual void symdefault (float def) { }

    /// Add a default string value to the last declared symbol.
    ///
    virtual void symdefault (const char *def) { }

    /// Add a hint.
    ///
    virtual void hint (const char *string) { }

    /// New code section marker designating subsequent instructions.
    ///
    virtual void codemarker (const char *name) { }

    /// Add an instruction.
    ///
    virtual void instruction (int label, const char *opcode) { }

    /// Add an argument to the last instruction.
    ///
    virtual void instruction_arg (const char *name) { }

    /// Add a jump target to the last instruction.
    ///
    virtual void instruction_jump (int target) { }

    /// Pointer to the one and only lexer in effect.  This is 'public',
    /// but NOBODY should modify this except for this class and the
    /// lexer internals.
    static osoFlexLexer *osolexer;

    static OSOReader *osoreader;

private:
    static mutex m_osoread_mutex;
};



}; // namespace pvt
}; // namespace OSL


#endif /* OSOREADER_H */
