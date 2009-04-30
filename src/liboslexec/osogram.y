/** Parser for OpenShadingLanguage 'object' files
 **/

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


%{

// C++ declarations

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

#include "oslexec.h"
#include "oslexec_pvt.h"

#undef yylex
#define yyFlexLexer osoFlexLexer
#include "FlexLexer.h"

void yyerror (const char *err);
#define yylex OSL::ShadingSystem::osolexer->yylex

using namespace OSL;
using namespace OSL::pvt;


osoFlexLexer *ShadingSystem::osolexer = NULL;



// Convert from the lexer's symbolic type (COLORTYPE, etc.) to a TypeDesc.
inline TypeDesc
lextype (int lex)
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
    case VOIDTYPE   : return TypeDesc::VOID;
    default: return PT_UNKNOWN;
    }
}


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
%type <i> oso_file version shader_declaration shader_type
%type <i> symbols_opt symbols symbol typespec simple_typename arraylen_opt
%type <i> initial_val_opt initial_val
%type <i> initial_floats initial_float initial_strings initial_string
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
                    $$ = 0;
                }
	;

version
        : IDENTIFIER FLOAT_LITERAL ENDOFLINE
                {
                    std::cerr << "Recognized version " << $2 << "\n";
                    $$ = 0;
                }
        ;

shader_declaration
        : shader_type IDENTIFIER hints_opt ENDOFLINE
                {
                    std::cerr << "Recognized shader_declaration\n";
                }
        ;

symbols_opt
        : symbols                       { $$ = 0; }
        | /* empty */                   { $$ = 0; }
        ;

codemarker
        : CODE IDENTIFIER ENDOFLINE
                {
                    std::cerr << "Recognized code marker  " << $2 << "\n";
                }
        ;

instructions
        : instruction
        | instructions instruction
        ;

instruction
        : label opcode arguments_opt jumptargets_opt hints_opt ENDOFLINE
                {
                    std::cerr << "Recognized instruction " << $2 << "\n";
                }
        | codemarker
        | ENDOFLINE
        ;

shader_type
        : IDENTIFIER                    { $$ = 0; }
        ;

symbols
        : symbol
        | symbols symbol
        ;

symbol
        : SYMTYPE typespec IDENTIFIER arraylen_opt initial_val_opt hints_opt ENDOFLINE
                {
                    std::cerr << "Recognized symbol " << $3 << "\n";
                }
        | ENDOFLINE
        ;

/* typespec operates by merely setting the current_typespec */
typespec
        : simple_typename
                {
                    $$ = 0;
                }
        | CLOSURE simple_typename
                {
                    $$ = 0;
                }
        | STRUCT IDENTIFIER /* struct name */
                {
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

initial_val_opt
        : initial_val
        | /* empty */                   { $$ = 0; }
        ;

initial_val
        : initial_floats
        | initial_strings
        ;

initial_floats
        : initial_float
        | initial_floats initial_float
        ;

initial_strings
        : initial_string
        | initial_strings initial_string
        ;

initial_float
        : FLOAT_LITERAL                 { $$ = 0; }
        | INT_LITERAL                   { $$ = 0; }
        ;

initial_string
        : STRING_LITERAL                { $$ = 0; }
        ;

label
        : INT_LITERAL ':'
        | /* empty */                   { $$ = 0; }
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
        : IDENTIFIER                    { $$ = 0; }
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
        : HINT                          { $$ = 0; }
        ;

%%



void
yyerror (const char *err)
{
//    oslcompiler->error (oslcompiler->filename(), oslcompiler->lineno(),
//                        "Syntax error: %s", err);
    fprintf (stderr, "Error, line %d: %s", 
             OSL::ShadingSystem::osolexer->lineno(), err);
}


