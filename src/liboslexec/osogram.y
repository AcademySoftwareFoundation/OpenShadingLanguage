/* Copyright Contributors to the Open Shading Language project.
 * SPDX-License-Identifier: BSD-3-Clause
 * https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
 */

/** Parser for OpenShadingLanguage 'object' files
 **/

%define api.pure full
%lex-param   { void *scanner }
%lex-param   { OSL::pvt::OSOReader *osoreader }
%parse-param { void *scanner }
%parse-param { OSL::pvt::OSOReader *osoreader }

%{

// C++ declarations

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

#include "osoreader.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wparentheses-equality"
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif

%}


// This is the definition for the union that defines YYSTYPE
%union
{
    int         i;  // For integer falues
    float       f;  // For float values
    const char *s;  // For string values -- guaranteed to be a ustring.c_str()
}

%{
OSL_NAMESPACE_ENTER
namespace pvt {

int yylex(YYSTYPE* yylval_param, YYLTYPE* yylloc_param, void* yyscanner, OSOReader* osoreader);
void yyerror (YYLTYPE* yylloc_param, void* yyscanner, OSOReader* osoreader, const char* err);

TypeDesc osolextype (int lex);

} // namespace pvt
OSL_NAMESPACE_EXIT

using namespace OSL;
using namespace OSL::pvt;
%}

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
                    osoreader->codeend ();
                    $$ = 0;
                }
	;

version
        : IDENTIFIER FLOAT_LITERAL ENDOFLINE
                {
                    int major = (int) $2;
                    int minor = (int) (100*($2-major) + 0.5);
                    osoreader->version ($1, major, minor);
                    $$ = 0;
                }
        ;

shader_declaration
        : shader_type IDENTIFIER 
                {
                    osoreader->shader ($1, $2);
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
                    if (! osoreader->parse_code_section())
                        YYACCEPT;
                    osoreader->codemarker ($2);
                }
        ;

instructions
        : instruction
        | instructions instruction
        ;

instruction
        : label opcode 
                {
                    osoreader->instruction ($1, $2);
                }
            arguments_opt jumptargets_opt hints_opt ENDOFLINE
                {
                    osoreader->instruction_end ();
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
                    if ((SymType)$1 == SymTypeTemp &&
                        osoreader->stop_parsing_at_temp_symbols())
                        YYACCEPT;
                    TypeSpec typespec = osoreader->current_typespec();
                    if ($3)
                        typespec.make_array ($3);
                    osoreader->symbol ((SymType)$1, typespec, $4);
                }
            initial_values_opt hints_opt
                {
                    osoreader->parameter_done ();
                }
            ENDOFLINE
        | ENDOFLINE
        ;

/* typespec operates by merely setting the current_typespec */
typespec
        : simple_typename
                {
                    osoreader->current_typespec(osolextype ($1));
                    $$ = 0;
                }
        | CLOSURE simple_typename
                {
                    osoreader->current_typespec(TypeSpec (osolextype ($2), true));
                    $$ = 0;
                }
        | STRUCT IDENTIFIER
                {
                    osoreader->current_typespec(TypeSpec ($2, 0));
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
        | '[' ']'                       { $$ = -1; }
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
                    osoreader->symdefault ($1);
                    $$ = 0;
                }
        | INT_LITERAL
                {
                    osoreader->symdefault ($1);
                    $$ = 0;
                }
        | STRING_LITERAL
                {
                    osoreader->symdefault ($1);
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
                    osoreader->instruction_arg ($1);
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
                    osoreader->instruction_jump ($1);
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
                    osoreader->hint ($1);
                    $$ = 0;
                }
        ;


%%



void
OSL::pvt::yyerror (YYLTYPE* yylloc_param, void* yyscanner, OSOReader* osoreader, const char* err)
{
    osoreader->errhandler().error ("Error, line %d: %s", 
             osoreader->lineno(), err);
}






// Convert from the lexer's symbolic type (COLORTYPE, etc.) to a TypeDesc.
inline TypeDesc
OSL::pvt::osolextype (int lex)
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
    default: return TypeDesc::UNKNOWN;
    }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
