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

/** Parser for OpenShadingLanguage 'object' files
 **/


%{

// C++ declarations

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

#include "osoreader.h"

#undef yylex
#define yyFlexLexer osoFlexLexer
#include "FlexLexer.h"

using namespace OSL;
using namespace OSL::pvt;

void yyerror (const char *err);

#define yylex OSOReader::osolexer->yylex
#define reader OSOReader::reader

static TypeSpec current_typespec;
static std::string current_shader_name;

// Forward declaration
OSL_NAMESPACE_ENTER
namespace pvt {
TypeDesc lextype (int lex);
};
OSL_NAMESPACE_EXIT

%}


// This is the definition for the union that defines YYSTYPE
%union
{
    int         i;  // For integer falues
    float       f;  // For float values
    const char *s;  // For string values -- guaranteed to be a ustring.c_str()
}


// Tell Bison to track locations for improved error messages
%locations


// Define the terminal symbols.
%token <s> IDENTIFIER STRING_LITERAL HINT
%token <i> INT_LITERAL
%token <f> FLOAT_LITERAL
%token <i> COLORTYPE FLOATTYPE INTTYPE MATRIXTYPE 
%token <i> NORMALTYPE POINTTYPE STRINGTYPE VECTORTYPE VOIDTYPE CLOSURE STRUCT
%token <i> CODE SYMTYPE ENDOFLINE

// Define the nonterminals 
%type <i> oso_file version shader_declaration
%type <s> shader_type
%type <i> symbols_opt symbols symbol typespec simple_typename arraylen_opt
%type <i> initial_values_opt initial_values initial_value
%type <i> codemarker label
%type <i> instructions instruction
%type <s> opcode
%type <i> arguments_opt arguments argument
%type <i> jumptargets_opt jumptargets jumptarget
%type <i> hints_opt hints hint

// Define the starting nonterminal
%start oso_file


%%

oso_file
        : version shader_declaration symbols_opt codemarker instructions
                {
                    OSOReader::osoreader->codeend ();
                    $$ = 0;
                }
	;

version
        : IDENTIFIER FLOAT_LITERAL ENDOFLINE
                {
                    int major = (int) $2;
                    int minor = (int) (100*($2-major) + 0.5);
                    OSOReader::osoreader->version ($1, major, minor);
                    $$ = 0;
                }
        ;

shader_declaration
        : shader_type IDENTIFIER 
                {
                    OSOReader::osoreader->shader ($1, $2);
                    current_shader_name = $2;
                }
            hints_opt ENDOFLINE
                {
                    $$ = 0;
                }
        ;

symbols_opt
        : symbols                       { $$ = 0; }
        | /* empty */                   { $$ = 0; }
        ;

codemarker
        : CODE IDENTIFIER ENDOFLINE
                {
                    OSOReader::osoreader->codemarker ($2);
                }
        ;

instructions
        : instruction
        | instructions instruction
        ;

instruction
        : label opcode 
                {
                    OSOReader::osoreader->instruction ($1, $2);
                }
            arguments_opt jumptargets_opt hints_opt ENDOFLINE
                {
                    OSOReader::osoreader->instruction_end ();
                }
        | codemarker
        | ENDOFLINE
        ;

shader_type
        : IDENTIFIER
        ;

symbols
        : symbol
        | symbols symbol
        ;

symbol
        : SYMTYPE typespec arraylen_opt IDENTIFIER 
                {
                    TypeSpec typespec = current_typespec;
                    if ($3)
                        typespec.make_array ($3);
                    OSOReader::osoreader->symbol ((SymType)$1, typespec, $4);
                }
            initial_values_opt hints_opt ENDOFLINE
        | ENDOFLINE
        ;

/* typespec operates by merely setting the current_typespec */
typespec
        : simple_typename
                {
                    current_typespec = lextype ($1);
                    $$ = 0;
                }
        | CLOSURE simple_typename
                {
                    current_typespec = TypeSpec (lextype ($2), true);
                    $$ = 0;
                }
        | STRUCT IDENTIFIER
                {
                    // Prepend the shader name to make globally unique
                    std::string mangled = current_shader_name + "_" + $2;
                    current_typespec = TypeSpec (mangled.c_str(), 0);
                    $$ = 0;
                }
        ;

simple_typename
        : COLORTYPE
        | FLOATTYPE
        | INTTYPE
        | MATRIXTYPE
        | NORMALTYPE
        | POINTTYPE
        | STRINGTYPE
        | VECTORTYPE
        | VOIDTYPE
        ;

arraylen_opt
        : '[' INT_LITERAL ']'           { $$ = $2; }
        | /* empty */                   { $$ = 0; }
        ;

initial_values_opt
        : initial_values
        | /* empty */                   { $$ = 0; }
        ;

initial_values
        : initial_value
        | initial_values initial_value
        ;

initial_value
        : FLOAT_LITERAL
                {
                    OSOReader::osoreader->symdefault ($1);
                    $$ = 0;
                }
        | INT_LITERAL
                {
                    OSOReader::osoreader->symdefault ($1);
                    $$ = 0;
                }
        | STRING_LITERAL
                {
                    OSOReader::osoreader->symdefault ($1);
                    $$ = 0;
                }
        ;

label
        : INT_LITERAL ':'
        | /* empty */                   { $$ = -1; }
        ;

opcode
        : IDENTIFIER
        ;

arguments_opt
        : arguments
        | /* empty */                   { $$ = 0; }
        ;

arguments
        : argument
        | arguments argument
        ;

argument
        : IDENTIFIER
                {
                    OSOReader::osoreader->instruction_arg ($1);
                }
        ;

jumptargets_opt
        : jumptargets
        | /* empty */                   { $$ = 0; }
        ;

jumptargets
        : jumptarget
        | jumptargets jumptarget
        ;

jumptarget
        : INT_LITERAL
                {
                    OSOReader::osoreader->instruction_jump ($1);
                }
        ;

hints_opt
        : hints
        | /* empty */                   { $$ = 0; }
        ;

hints
        : hint
        | hints hint
        ;

hint
        : HINT
                {
                    OSOReader::osoreader->hint ($1);
                }
        ;

%%



void
yyerror (const char *err)
{
//    oslcompiler->error (oslcompiler->filename(), oslcompiler->lineno(),
//                        "Syntax error: %s", err);
    fprintf (stderr, "Error, line %d: %s", 
             OSOReader::osoreader->lineno(), err);
}






// Convert from the lexer's symbolic type (COLORTYPE, etc.) to a TypeDesc.
TypeDesc
OSL::pvt::lextype (int lex)
{
    switch (lex) {
    case COLORTYPE  : return TypeDesc::TypeColor;
    case FLOATTYPE  : return TypeDesc::TypeFloat;
    case INTTYPE    : return TypeDesc::TypeInt;
    case MATRIXTYPE : return TypeDesc::TypeMatrix;
    case NORMALTYPE : return TypeDesc::TypeNormal;
    case POINTTYPE  : return TypeDesc::TypePoint;
    case STRINGTYPE : return TypeDesc::TypeString;
    case VECTORTYPE : return TypeDesc::TypeVector;
    case VOIDTYPE   : return TypeDesc::NONE;
    default: return PT_UNKNOWN;
    }
}
