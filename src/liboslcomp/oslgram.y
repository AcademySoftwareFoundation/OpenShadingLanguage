/** Parser for Sony Imageworks Shading Language
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

#include "ustring.h"
#include "oslcomp_pvt.h"

#undef yylex
#define yyFlexLexer oslFlexLexer
#include "FlexLexer.h"

void yyerror (const char *err);
#define yylex oslcompiler->lexer()->yylex

using namespace OSL::pvt;

%}


// This is the definition for the union that defines YYSTYPE
%union
{
    int         i;  // For integer falues
    float       f;  // For float values
    ASTNode    *n;  // Abstract Syntax Tree node
    const char *s;  // For string values -- guaranteed to be a ustring.c_str()
}


// Tell Bison to track locations for improved error messages
%locations


// Define the terminal symbols.
%token <s> IDENTIFIER STRING_LITERAL
%token <i> INTLITERAL
%token <f> FLOAT_LITERAL
%token <i> COLOR FLOAT INT MATRIX NORMAL POINT STRING VECTOR VOID
%token <i> CLOSURE OUTPUT PUBLIC STRUCT
%token <i> BREAK CONTINUE DO ELSE FOR IF ILLUMINATE ILLUMINANCE RETURN WHILE
%token <i> RESERVED


// Define the nonterminals 
%type <n> shader_file 
%type <n> global_declarations_opt global_declarations global_declaration


// Define operator precedence, lowest-to-highest
%left <i> ','
%right <i> '=' ADD_ASSIGN SUB_ASSIGN MUL_ASSIGN DIV_ASSIGN BIT_AND_ASSIGN BIT_OR_ASSIGN BIT_XOR_ASSIGN SHL_ASSIGN SHR_ASSIGN
%right <i> '?' ':'
%left <i> LOGIC_OR_OP
%left <i> LOGIC_AND_OP
%left <i> '|'
%left <i> '^'
%left <i> '&'
%left <i> EQ_OP NE_OP
%left <i> '>' GE_OP '<' LE_OP
%left <i> SHL_OP SHR_OP
%left <i> '+' '-'
%left <i> '*' '/' '%'
%right <i> UMINUS_PREC '!' '~'
%right <i> INCREMENT DECREMENT
%left <i> '(' ')'
%left <i> '[' ']'
%left <i> META_BEGIN


// Define the starting nonterminal
%start shader_file


%%

shader_file : global_declarations_opt shader_declaration
	;

global_declarations_opt
        : global_declarations
        | /* empty */
        ;

global_declarations
        : global_declaration
        | global_declarations global_declaration
        ;

global_declaration
        : function_declaration
        | struct_declaration
        ;

%%



void
yyerror (const char *err)
{
    oslcompiler->error (err);
}


