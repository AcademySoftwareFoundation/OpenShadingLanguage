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
#include "symtab.h"

#undef yylex
#define yyFlexLexer oslFlexLexer
#include "FlexLexer.h"

void yyerror (const char *err);
#define yylex oslcompiler->lexer()->yylex

using namespace OSL::pvt;



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
    ASTNode    *n;  // Abstract Syntax Tree node
    const char *s;  // For string values -- guaranteed to be a ustring.c_str()
}


// Tell Bison to track locations for improved error messages
%locations


// Define the terminal symbols.
%token <s> IDENTIFIER STRING_LITERAL
%token <i> INT_LITERAL
%token <f> FLOAT_LITERAL
%token <i> COLORTYPE FLOATTYPE INTTYPE MATRIXTYPE 
%token <i> NORMALTYPE POINTTYPE STRINGTYPE VECTORTYPE VOIDTYPE
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
%type <i> unary_op incdec_op incdec_op_opt
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
        : function_declaration          { $$ = 0;  /* FIXME */ }
        | struct_declaration            { $$ = 0;  /* FIXME */ }
        | shader_declaration
        ;

shader_declaration
        : shadertype IDENTIFIER metadata_block_opt '(' shader_formal_params_opt ')' '{' statement_list '}'
                {
                    $$ = new ASTshader_declaration (oslcompiler, $1,
                                                    ustring($2), $5, $8, $3);
                    if (oslcompiler->shader_is_defined()) {
                        oslcompiler->error (oslcompiler->filename(),
                                            oslcompiler->lineno(),
                                            "Only one shader is allowed per file.");
                        delete $$;
                        $$ = NULL;
                    } else {
                        oslcompiler->shader ($$);
                    }
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
        : outputspec typespec IDENTIFIER initializer metadata_block_opt { $$ = 0; /*FIXME*/ }
        | outputspec typespec IDENTIFIER arrayspec array_initializer metadata_block_opt { $$ = 0; /*FIXME*/ }
        ;

metadata_block_opt
        : METADATA_BEGIN metadata ']' ']' { $$ = 0;  /*FIXME*/ }
        | /* empty */                   { $$ = 0; }
        ;

metadata
        : metadatum
        | metadata ',' metadatum        { $$ = concat ($1, $3); }
        ;

metadatum
        : simple_typename IDENTIFIER initializer { $$ = 0;  /*FIXME*/ }
        | simple_typename IDENTIFIER arrayspec array_initializer { $$ = 0;  /*FIXME*/ }
        ;

function_declaration
        : typespec IDENTIFIER '(' function_formal_params_opt ')' '{' statement_list '}'
                {
                    $$ = 0; /*FIXME*/
                }
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
        : outputspec typespec IDENTIFIER
                {
                    ASTvariable_declaration *var;
                    var = new ASTvariable_declaration (oslcompiler,
                                              oslcompiler->current_typespec(),
                                              ustring ($3), NULL, true);
                    var->make_output ($1);
                    $$ = var;
                }
        | outputspec typespec IDENTIFIER arrayspec
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    TypeDesc simple = t.type();
                    simple.arraylen = $4;
                    t = TypeSpec (simple, t.is_closure());
                    // FIXME -- won't work for a struct
                    ASTvariable_declaration *var;
                    var = new ASTvariable_declaration (oslcompiler, t, 
                                                       ustring($3), NULL, true);
                    var->make_output ($1);
                    $$ = var;
                }
        ;

struct_declaration
        : STRUCT IDENTIFIER '{' field_declarations '}' { $$ = 0;  /*FIXME*/ }
        ;

field_declarations
        : field_declaration
        | field_declarations field_declaration  { $$ = concat ($1, $2) }
        ;

field_declaration
        : typespec IDENTIFIER ';'               { $$ = 0; /*FIXME*/ }
        | typespec IDENTIFIER arrayspec ';'     { $$ = 0; /*FIXME*/ }
        ;

local_declaration
        : function_declaration
        | variable_declaration
        ;

variable_declaration
        : typespec def_expressions ';'          { $$ = $2; }
        ;

def_expressions
        : def_expression
        | def_expressions ',' def_expression    { $$ = concat ($1, $3) }
        ;

def_expression
        : IDENTIFIER initializer_opt
                {
                    $$ = new ASTvariable_declaration (oslcompiler,
                                              oslcompiler->current_typespec(),
                                              ustring($1), $2);
                }
        | IDENTIFIER arrayspec array_initializer_opt
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    TypeDesc simple = t.type();
                    simple.arraylen = $2;
                    t = TypeSpec (simple, t.is_closure());
                    // FIXME -- won't work for a struct
                    $$ = new ASTvariable_declaration (oslcompiler, t, 
                                                      ustring($1), $3);
                }
        ;

initializer_opt
        : initializer
        | /* empty */                   { $$ = 0; }
        ;

initializer
        : '=' expression                { $$ = $2; }
        ;

array_initializer_opt
        : array_initializer
        | /* empty */                   { $$ = 0; }
        ;

array_initializer
        : '=' '{' expression_list '}'   { $$ = $3; }
        ;

shadertype
        : IDENTIFIER
                {
                    if (! strcmp ($1, "shader"))
                        $$ = OSL::ShadTypeGeneric;
                    else if (! strcmp ($1, "surface"))
                        $$ = OSL::ShadTypeSurface;
                    else if (! strcmp ($1, "displacement"))
                        $$ = OSL::ShadTypeDisplacement;
                    else if (! strcmp ($1, "volume"))
                        $$ = OSL::ShadTypeVolume;
                    else if (! strcmp ($1, "light"))
                        $$ = OSL::ShadTypeLight;
                    else {
                        oslcompiler->error (oslcompiler->filename(),
                                            oslcompiler->lineno(),
                                            "Unknown shader type: %s", $1);
                        $$ = OSL::ShadTypeUnknown;
                    }
                }
        ;

/* outputspec operates by merely setting the current_output to whether
 * or not we're declaring an output parameter.
 */
outputspec
        : OUTPUT                { oslcompiler->current_output (true); $$ = 1; }
        | /* empty */           { oslcompiler->current_output (false); $$ = 0; }
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

arrayspec
        : '[' INT_LITERAL ']'           { $$ = $2; }
        ;

/* typespec operates by merely setting the current_typespec */
typespec
        : simple_typename
                {
                    oslcompiler->current_typespec (TypeSpec (lextype ($1)));
                    $$ = 0;
                }
        | simple_typename CLOSURE
                {
                    oslcompiler->current_typespec (TypeSpec (lextype ($1), true));
                    $$ = 0;
                }
        | IDENTIFIER /* struct name */  { $$ = 0; /*FIXME*/ }
        ;

statement_list
        : statement statement_list      { $$ = concat ($1, $2); }
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
        : IF '(' expression ')' statement
                {
                    $$ = new ASTconditional_statement (oslcompiler, $3, $5);
                }
        | IF '(' expression ')' statement ELSE statement
                {
                    $$ = new ASTconditional_statement (oslcompiler, $3, $5, $7);
                }
        ;

loop_statement
        : WHILE '(' expression ')' statement
                {
                    $$ = new ASTloop_statement (oslcompiler,
                                                ASTloop_statement::LoopWhile,
                                                NULL, $3, NULL, $5);
                }
        | DO statement WHILE '(' expression ')' ';'
                {
                    $$ = new ASTloop_statement (oslcompiler,
                                                ASTloop_statement::LoopDo,
                                                NULL, $5, NULL, $2);
                }
        | FOR '(' for_init_statement expression_opt ';' expression_opt ')' statement
                {
                    $$ = new ASTloop_statement (oslcompiler,
                                                ASTloop_statement::LoopFor,
                                                $3, $4, $6, $8);
                }
        ;

loopmod_statement
        : BREAK ';'
                {
                    $$ = new ASTloopmod_statement (oslcompiler, ASTloopmod_statement::LoopModBreak);
                }
        | CONTINUE ';'
                {
                    $$ = new ASTloopmod_statement (oslcompiler, ASTloopmod_statement::LoopModContinue);
                }
        ;

return_statement
        : RETURN expression_opt ';'
                {
                    $$ = new ASTreturn_statement (oslcompiler, $2);
                }
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
        : INT_LITERAL           { $$ = new ASTliteral (oslcompiler, $1); }
        | FLOAT_LITERAL         { $$ = new ASTliteral (oslcompiler, $1); }
        | STRING_LITERAL        { $$ = new ASTliteral (oslcompiler, ustring($1)); }
        | variable_ref
        | incdec_op variable_lvalue 
                {
                    DASSERT ($2->nodetype() == ASTNode::variable_ref_node);
                    ((ASTvariable_ref *)$2)->add_preop ($1);
                    $$ = $2;
                }
        | binary_expression
        | unary_op expression %prec UMINUS_PREC
                {
                    $$ = new ASTunary_expression (oslcompiler, $1, $2);
                }
        | '(' expression ')'                    { $$ = $2; }
        | function_call
        | assign_expression
        | ternary_expression
        | typecast_expression
        | type_constructor
        ;

variable_lvalue
        : IDENTIFIER array_deref_opt component_deref_opt component_deref_opt 
                {
                    $$ = new ASTvariable_ref (oslcompiler, ustring($1),
                                              $2, $3, $4);
                }
        ;

variable_ref
        : variable_lvalue incdec_op_opt
                {
                    DASSERT ($1->nodetype() == ASTNode::variable_ref_node);
                    ((ASTvariable_ref *)$1)->add_postop ($2);
                    $$ = $1;
                }
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
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::LogicalOr, $1, $3);
                }
        | expression LOGIC_AND_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::LogicalAnd, $1, $3);
                }
        | expression '|' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::BitwiseOr, $1, $3);
                }
        | expression '^' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::BitwiseXor, $1, $3);
                }
        | expression '&' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::BitwiseAnd, $1, $3);
                }
        | expression EQ_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Equal, $1, $3);
                }
        | expression NE_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::NotEqual, $1, $3);
                }
        | expression '>' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Greater, $1, $3);
                }
        | expression GE_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::GreaterEqual, $1, $3);
                }
        | expression '<' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Less, $1, $3);
                }
        | expression LE_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::LessEqual, $1, $3);
                }
        | expression SHL_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::ShiftLeft, $1, $3);
                }
        | expression SHR_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::ShiftRight, $1, $3);
                }
        | expression '+' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Add, $1, $3);
                }
        | expression '-' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Sub, $1, $3);
                }
        | expression '*' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Mul, $1, $3);
                }
        | expression '/' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Div, $1, $3);
                }
        | expression '%' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTbinary_expression::Mod, $1, $3);
                }
        ;

unary_op
        : '-' | '+' | '!' | '~'
        ;

incdec_op_opt
        : incdec_op
        | /* empty */                   { $$ = 0; }
        ;

incdec_op
        : INCREMENT                     { $$ = +1; }
        | DECREMENT                     { $$ = -1; }
        ;

type_constructor
        : typespec '(' expression_list ')'
                {
                    $$ = new ASTtypecast_expression (oslcompiler, 
                                                     oslcompiler->current_typespec(),
                                                     $3);
                }
        ;

function_call
        : IDENTIFIER '(' function_args_opt ')'
                {
                    $$ = new ASTfunction_call (oslcompiler, $1, $3);
                }
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
                { $$ = new ASTassign_expression (oslcompiler, $1,
                                ASTassign_expression::Assign, $3); }
        | variable_lvalue MUL_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
                	        ASTassign_expression::MulAssign, $3); }
        | variable_lvalue DIV_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::DivAssign, $3); }
        | variable_lvalue ADD_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::AddAssign, $3); }
        | variable_lvalue SUB_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::SubAssign, $3); }
        | variable_lvalue BIT_AND_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::BitwiseAndAssign, $3); }
        | variable_lvalue BIT_OR_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::BitwiseOrAssign, $3); }
        | variable_lvalue BIT_XOR_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::BitwiseXorAssign, $3); }
        | variable_lvalue SHL_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::ShiftLeftAssign, $3); }
        | variable_lvalue SHR_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTassign_expression::ShiftRightAssign, $3); }
        ;

ternary_expression
        : expression '?' expression ':' expression
                {
                    $$ = new ASTternary_expression (oslcompiler, $1, $3, $5);
                }
        ;

typecast_expression
        : '(' simple_typename ')' expression
                {
                    $$ = new ASTtypecast_expression (oslcompiler, 
                                                     TypeSpec (lextype ($2)),
                                                     $4);
                }
        ;

%%



void
yyerror (const char *err)
{
    oslcompiler->error (oslcompiler->filename(), oslcompiler->lineno(),
                        "Syntax error: %s", err);
}


