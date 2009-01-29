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

#include "oslcomp_pvt.h"
#include "ast.h"

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
%token <i> INT_LITERAL
%token <f> FLOAT_LITERAL
%token <i> COLOR FLOAT INT MATRIX NORMAL POINT STRING VECTOR VOID
%token <i> CLOSURE OUTPUT PUBLIC STRUCT
%token <i> BREAK CONTINUE DO ELSE FOR IF ILLUMINATE ILLUMINANCE RETURN WHILE
%token <i> RESERVED


// Define the nonterminals 
%type <n> shader_file 
%type <n> global_declarations_opt global_declarations global_declaration
%type <n> shader_declaration 
%type <n> shader_formal_params_opt shader_formal_params shader_formal_param
%type <n> metadata_block_opt metadata metadatum
%type <n> function_declaration function_formal_params_opt 
%type <n> function_formal_params function_formal_param
%type <n> struct_declaration field_declarations field_declaration
%type <n> variable_declaration def_expressions def_expression
%type <n> initializer_opt initializer array_initializer_opt array_initializer 
%type <i> shadertype outputspec arrayspec simple_typename
%type <n> typespec 
%type <n> statement_list statement scoped_statements local_declaration
%type <n> conditional_statement loop_statement loopmod_statement
%type <n> return_statement
%type <n> for_init_statement
%type <n> expression_list expression_opt expression
%type <n> variable_lvalue variable_ref array_deref_opt component_deref_opt
%type <i> unary_op incdec_op 
%type <n> type_constructor function_call function_args_opt function_args
%type <n> assign_expression ternary_expression typecast_expression
%type <n> binary_expression

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
%left <i> METADATA_BEGIN


// Define the starting nonterminal
%start shader_file


%%

shader_file : global_declarations_opt
	;

global_declarations_opt
        : global_declarations
        | /* empty */                   { $$ = 0; }
        ;

global_declarations
        : global_declaration
        | global_declarations global_declaration    { $$ = concat ($1, $2); }
        ;

global_declaration
        : function_declaration
        | struct_declaration
        | shader_declaration
        ;

shader_declaration
        : shadertype IDENTIFIER metadata_block_opt '(' shader_formal_params_opt ')' '{' statement_list '}'
                {
                    $$ = new ASTshader_declaration (oslcompiler, 0 /* stype */,
                                                    ustring($2), $5, $8, $3);
                }
        ;

shader_formal_params_opt
        : shader_formal_params
        | /* empty */                   { $$ = 0; }
        ;

shader_formal_params
        : shader_formal_param
        | shader_formal_params ',' shader_formal_param
                {
                    $$ = concat ($1, $3);
                }
        ;

shader_formal_param
        : outputspec typespec IDENTIFIER initializer metadata_block_opt { $$ = 0; }
        | outputspec typespec IDENTIFIER arrayspec array_initializer metadata_block_opt { $$ = 0; }
        ;

metadata_block_opt
        : METADATA_BEGIN metadata ']' ']' { $$ = 0; }
        | /* empty */                   { $$ = 0; }
        ;

metadata
        : metadatum
        | metadata ',' metadatum        { $$ = concat ($1, $3); }
        ;

metadatum
        : simple_typename IDENTIFIER initializer { $$ = 0; }
        | simple_typename IDENTIFIER arrayspec array_initializer { $$ = 0; }
        ;

function_declaration
        : typespec IDENTIFIER '(' function_formal_params_opt ')' '{' statement_list '}'
        ;

function_formal_params_opt
        : function_formal_params
        | /* empty */                   { $$ = 0; }
        ;

function_formal_params
        : function_formal_param
        | function_formal_params ',' function_formal_param
                {
                    $$ = concat ($1, $3);
                }
        ;

function_formal_param
        : outputspec typespec IDENTIFIER           { $$ = 0; }
        | outputspec typespec IDENTIFIER arrayspec { $$ = 0; }
        ;

struct_declaration
        : STRUCT IDENTIFIER '{' field_declarations '}' { $$ = 0; }
        ;

field_declarations
        : field_declaration
        | field_declarations field_declaration  { $$ = concat ($1, $2) }
        ;

field_declaration
        : typespec IDENTIFIER ';'
        | typespec IDENTIFIER arrayspec ';'
        ;

local_declaration
        : function_declaration
        | variable_declaration
        ;

variable_declaration
        : typespec def_expressions ';'
        ;

def_expressions
        : def_expression
        | def_expressions ',' def_expression    { $$ = concat ($1, $3) }
        ;

def_expression
        : IDENTIFIER initializer_opt { $$ = 0; }
        | IDENTIFIER arrayspec array_initializer_opt { $$ = 0; }
        ;

initializer_opt
        : initializer
        | /* empty */                   { $$ = 0; }
        ;

initializer
        : '=' expression { $$ = 0; }
        ;

array_initializer_opt
        : array_initializer
        | /* empty */                   { $$ = 0; }
        ;

array_initializer
        : '=' '{' expression_list '}' { $$ = 0; }
        ;

shadertype
        : IDENTIFIER { $$ = 0; }
        ;

outputspec
        : OUTPUT
        | /* empty */                   { $$ = 0; }
        ;

simple_typename
        : COLOR
        | FLOAT
        | INT
        | MATRIX
        | NORMAL
        | POINT
        | STRING
        | VECTOR
        | VOID
        ;

arrayspec
        : '[' INT_LITERAL ']'
        ;

typespec
        : simple_typename { $$ = 0; }
        | simple_typename CLOSURE { $$ = 0; }
        | IDENTIFIER /* struct name */ { $$ = 0; }
        ;

statement_list
        : statement statement_list
        | /* empty */                   { $$ = 0; }
        ;

statement
        : scoped_statements
        | conditional_statement
        | loop_statement
        | loopmod_statement
        | return_statement
        | local_declaration
        | expression ';'
        | ';'                           { $$ = 0; }
        ;

scoped_statements
        : '{' statement_list '}'        { $$ = $2; }
        ;

conditional_statement
        : IF '(' expression ')' statement { $$ = 0; }
        | IF '(' expression ')' statement ELSE statement { $$ = 0; }
        ;

loop_statement
        : WHILE '(' expression ')' statement { $$ = 0; }
        | DO statement WHILE '(' expression ')' ';' { $$ = 0; }
        | FOR '(' for_init_statement expression_opt ';' expression_opt ')' statement { $$ = 0; }
        ;

loopmod_statement
        : BREAK ';' { $$ = 0; }
        | CONTINUE ';' { $$ = 0; }
        ;

return_statement
        : RETURN expression_opt ';' { $$ = 0; }
        ;

for_init_statement
        : expression_opt ';'
        | variable_declaration
        ;

expression_list
        : expression
        | expression_list ',' expression        { $$ = concat ($1, $3) }
        ;

expression_opt
        : expression
        | /* empty */                   { $$ = 0; }
        ;

expression
        : INT_LITERAL { $$ = 0; }
        | FLOAT_LITERAL { $$ = 0; }
        | STRING_LITERAL { $$ = 0; }
        | incdec_op variable_ref { $$ = 0; }
        | binary_expression
        | unary_op expression %prec UMINUS_PREC { $$ = 0; }
        | '(' expression ')'                    { $$ = $2; }
        | function_call
        | assign_expression
        | ternary_expression
        | typecast_expression
        | type_constructor
        | variable_ref
        ;

variable_lvalue
        : IDENTIFIER array_deref_opt component_deref_opt component_deref_opt { $$ = 0; }
        ;

variable_ref
        : variable_lvalue incdec_op     { $$ = 0; }
        ;

array_deref_opt
        : '[' expression ']'            { $$ = $2; }
        | /* empty */                   { $$ = 0; }
        ;

component_deref_opt
        : '[' expression ']'            { $$ = $2; }
        | /* empty */                   { $$ = 0; }
        ;

binary_expression
        : expression LOGIC_OR_OP expression
        | expression LOGIC_AND_OP expression
        | expression '|' expression
        | expression '^' expression
        | expression '&' expression
        | expression EQ_OP expression
        | expression NE_OP expression
        | expression '>' expression
        | expression GE_OP expression
        | expression '<' expression
        | expression LE_OP expression
        | expression SHL_OP expression
        | expression SHR_OP expression
        | expression '+' expression
        | expression '-' expression
        | expression '*' expression
        | expression '/' expression
        | expression '%' expression
        ;

unary_op
        : '-' | '+' | '!' | '~'
        ;

incdec_op
        : INCREMENT | DECREMENT
        ;

type_constructor
        : typespec '(' expression_list ')' { $$ = 0; }
        ;

function_call
        : IDENTIFIER '(' function_args_opt ')' { $$ = 0; }
        ;

function_args_opt
        : function_args
        | /* empty */                   { $$ = 0; }
        ;

function_args
        : expression
        | function_args ',' expression          { $$ = concat ($1, $3) }
        ;

assign_expression
        : variable_lvalue '=' expression
        | variable_lvalue MUL_ASSIGN expression
        | variable_lvalue DIV_ASSIGN expression
        | variable_lvalue ADD_ASSIGN expression
        | variable_lvalue SUB_ASSIGN expression
        | variable_lvalue BIT_AND_ASSIGN expression
        | variable_lvalue BIT_OR_ASSIGN expression
        | variable_lvalue BIT_XOR_ASSIGN expression
        | variable_lvalue SHL_ASSIGN expression
        | variable_lvalue SHR_ASSIGN expression
        ;

ternary_expression
        : expression '?' expression ':' expression
        ;

typecast_expression
        : '(' simple_typename ')' expression { $$ = 0; }
        ;

%%



void
yyerror (const char *err)
{
    oslcompiler->error (err);
}


