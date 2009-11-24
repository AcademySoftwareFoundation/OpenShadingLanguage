/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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


/** Parser for Open Shading Language
 **/


%{

// C++ declarations

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
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


// Forward declaration
#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {
TypeDesc lextype (int lex);
};
};
#ifdef OSL_NAMESPACE
};
#endif

static std::stack<TypeSpec> typespec_stack; // just for function_declaration

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
%type <n> typed_field_list typed_field
%type <n> variable_declaration def_expressions def_expression
%type <n> initializer_opt initializer initializer_list_opt initializer_list
%type <i> shadertype outputspec arrayspec simple_typename
%type <n> typespec 
%type <n> statement_list statement scoped_statements local_declaration
%type <n> conditional_statement loop_statement loopmod_statement
%type <n> return_statement
%type <n> for_init_statement
%type <n> expression_list expression_opt expression
%type <n> id_or_field variable_lvalue variable_ref 
%type <i> unary_op incdec_op incdec_op_opt
%type <n> type_constructor function_call function_args_opt function_args
%type <n> assign_expression ternary_expression typecast_expression
%type <n> binary_expression
%type <s> string_literal_group

// Define operator precedence, lowest-to-highest
%left <i> ','
%right <i> '=' ADD_ASSIGN SUB_ASSIGN MUL_ASSIGN DIV_ASSIGN BIT_AND_ASSIGN BIT_OR_ASSIGN XOR_ASSIGN SHL_ASSIGN SHR_ASSIGN
%right <i> '?' ':'
%left <i> OR_OP
%left <i> AND_OP
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
        : function_declaration          { $$ = 0; }
        | struct_declaration            { $$ = 0; }
        | shader_declaration
        ;

shader_declaration
        : shadertype IDENTIFIER metadata_block_opt '(' shader_formal_params_opt ')' '{' statement_list '}'
                {
                    $$ = new ASTshader_declaration (oslcompiler, $1,
                                                    ustring($2), $5, $8, $3);
                    $$->sourceline (@2.first_line);
                    if (oslcompiler->shader_is_defined()) {
                        yyerror ("Only one shader is allowed per file.");
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
        : outputspec typespec IDENTIFIER initializer_opt metadata_block_opt
                {
                    ASTvariable_declaration *var;
                    TypeSpec t = oslcompiler->current_typespec();
                    var = new ASTvariable_declaration (oslcompiler,
                                                  t, ustring ($3), $4, true);
                    var->make_output ($1);
                    var->add_meta ($5);
                    $$ = var;
                    // Initializer is not really optional on a shader param,
                    // but try to give helpful error message
                    if ($4 == NULL) {
                        oslcompiler->error (oslcompiler->filename(),
                                            @3.first_line,
                                            "shader parameter '%s' MUST have a default initializer", $3);
                    }
                }
        | outputspec typespec IDENTIFIER arrayspec initializer_list metadata_block_opt
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    t.make_array ($4);
                    ASTvariable_declaration *var;
                    var = new ASTvariable_declaration (oslcompiler, t, 
                                                       ustring($3), $5, true);
                    var->make_output ($1);
                    var->add_meta ($6);
                    $$ = var;
                }
        | outputspec typespec IDENTIFIER initializer_list metadata_block_opt
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    if (! t.is_structure())
                        oslcompiler->error (oslcompiler->filename(),
                                            oslcompiler->lineno(),
                                            "Can't use '= {...}' initializer except with arrays or struct (%s)", $3);
                    ASTvariable_declaration *var;
                    var = new ASTvariable_declaration (oslcompiler, t,
                                                       ustring($3), $4, true);
                    var->make_output ($1);
                    var->add_meta ($5);
                    $$ = var;
                }
        ;

metadata_block_opt
        : METADATA_BEGIN metadata ']' ']'       { $$ = $2; }
        | /* empty */                           { $$ = 0; }
        ;

metadata
        : metadatum
        | metadata ',' metadatum        { $$ = concat ($1, $3); }
        ;

metadatum
        : simple_typename IDENTIFIER initializer
                {
                    ASTvariable_declaration *var;
                    var = new ASTvariable_declaration (oslcompiler, lextype($1),
                                                       ustring ($2), $3, false,
                                                       true /* ismeta */);
                    var->make_meta (true);
                    $$ = var;
                }
        | simple_typename IDENTIFIER arrayspec initializer_list
                {
                    TypeDesc simple = lextype ($1);
                    simple.arraylen = $3;
                    TypeSpec t (simple, t.is_closure());
                    ASTvariable_declaration *var;
                    var = new ASTvariable_declaration (oslcompiler, t, 
                                                       ustring ($2), $4, false,
                                                       true /* ismeata */);
                    var->make_meta (true);
                    $$ = var;
                }
        ;

function_declaration
        : typespec IDENTIFIER 
                {
                    oslcompiler->symtab().push ();  // new scope
                    typespec_stack.push (oslcompiler->current_typespec());
                }
          '(' function_formal_params_opt ')' '{' statement_list '}'
                {
                    oslcompiler->symtab().pop ();  // restore scope
                    $$ = new ASTfunction_declaration (oslcompiler,
                                                      typespec_stack.top(),
                                                      ustring($2), $5, $8, NULL);
                    typespec_stack.pop ();
                    // FIXME -- funcs don't have metadata. Should they?
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
                                              ustring ($3), NULL);
                    var->make_output ($1);
                    $$ = var;
                }
        | outputspec typespec IDENTIFIER arrayspec
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    t.make_array ($4);
                    ASTvariable_declaration *var;
                    var = new ASTvariable_declaration (oslcompiler, t, 
                                                       ustring($3), NULL);
                    var->make_output ($1);
                    $$ = var;
                }
        ;

struct_declaration
        : STRUCT IDENTIFIER '{' 
                {
                    ustring name ($2);
                    Symbol *s = oslcompiler->symtab().clash (name);
                    if (s) {
                        oslcompiler->error (oslcompiler->filename(), 
                                            oslcompiler->lineno(), 
                                            "\"%s\" already declared in this scope",
                                            name.c_str());
                        // FIXME -- print the file and line of the other definition
                    }
                    if (name[0] == '_' && name[1] == '_' && name[2] == '_') {
                        oslcompiler->error (oslcompiler->filename(), 
                            oslcompiler->lineno(),
                            "\"%s\" : sorry, can't start with three underscores",
                            name.c_str());
                    }
                    oslcompiler->symtab().new_struct (name);
                }
          field_declarations '}' ';'
                {
                    $$ = 0;
                }
        ;

field_declarations
        : field_declaration
        | field_declarations field_declaration  { $$ = concat ($1, $2) }
        ;

field_declaration
        : typespec typed_field_list ';'
        ;

typed_field_list
        : typed_field
        | typed_field_list ',' typed_field
        ;

typed_field
        : IDENTIFIER
                {
                    TypeSpec t = oslcompiler->current_typespec();
                    oslcompiler->symtab().add_struct_field (t, ustring($1));
                    $$ = 0;
                }
        | IDENTIFIER arrayspec
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    t.make_array ($2);
                    oslcompiler->symtab().add_struct_field (t, ustring($1));
                    $$ = 0;
                }
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
                    TypeSpec t = oslcompiler->current_typespec();
                    $$ = new ASTvariable_declaration (oslcompiler,
                                                      t, ustring($1), $2);
                }
        | IDENTIFIER arrayspec initializer_list_opt
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    t.make_array ($2);
                    $$ = new ASTvariable_declaration (oslcompiler, t, 
                                                      ustring($1), $3);
                }
        | IDENTIFIER initializer_list
                {
                    TypeSpec t = oslcompiler->current_typespec();
                    if (! t.is_structure())
                        oslcompiler->error (oslcompiler->filename(),
                                            oslcompiler->lineno(),
                                            "Can't use '= {...}' initializer except with arrays or struct (%s)", $1);
                    $$ = new ASTvariable_declaration (oslcompiler,
                                                      t, ustring($1), $2);
                }
        ;

initializer_opt
        : initializer
        | /* empty */                   { $$ = 0; }
        ;

initializer
        : '=' expression                { $$ = $2; }
        ;

initializer_list_opt
        : initializer_list
        | /* empty */                   { $$ = 0; }
        ;

initializer_list
        : '=' '{' expression_list '}'   { $$ = $3; }
        ;

shadertype
        : IDENTIFIER
                {
                    if (! strcmp ($1, "shader"))
                        $$ = ShadTypeGeneric;
                    else if (! strcmp ($1, "surface"))
                        $$ = ShadTypeSurface;
                    else if (! strcmp ($1, "displacement"))
                        $$ = ShadTypeDisplacement;
                    else if (! strcmp ($1, "volume"))
                        $$ = ShadTypeVolume;
                    // else if (! strcmp ($1, "light"))
                    //    $$ = ShadTypeLight;
                    else {
                        oslcompiler->error (oslcompiler->filename(),
                                            oslcompiler->lineno(),
                                            "Unknown shader type: %s", $1);
                        $$ = ShadTypeUnknown;
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
        | CLOSURE simple_typename
                {
                    oslcompiler->current_typespec (TypeSpec (lextype ($2), true));
                    $$ = 0;
                }
        | IDENTIFIER /* struct name */
                {
                    ustring name ($1);
                    Symbol *s = oslcompiler->symtab().find (name);
                    if (s && s->is_structure())
                        oslcompiler->current_typespec (TypeSpec ("", s->typespec().structure()));
                    else {
                        oslcompiler->current_typespec (TypeSpec (TypeDesc::UNKNOWN));
                        oslcompiler->error (oslcompiler->filename(),
                                            oslcompiler->lineno(),
                                            "Unknown struct name: %s", $1);
                    }
                    $$ = 0;
                }
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
        : '{' 
                {
                    oslcompiler->symtab().push ();  // new scope
                }
          statement_list '}'
                {
                    oslcompiler->symtab().pop ();  // restore scope
                    $$ = $3;
                }
        ;

conditional_statement
        : IF '(' expression ')' statement
                {
                    $$ = new ASTconditional_statement (oslcompiler, $3, $5);
                    $$->sourceline (@1.first_line);
                }
        | IF '(' expression ')' statement ELSE statement
                {
                    $$ = new ASTconditional_statement (oslcompiler, $3, $5, $7);
                    $$->sourceline (@1.first_line);
                }
        ;

loop_statement
        : WHILE '(' expression ')' statement
                {
                    $$ = new ASTloop_statement (oslcompiler,
                                                ASTloop_statement::LoopWhile,
                                                NULL, $3, NULL, $5);
                    $$->sourceline (@1.first_line);
                }
        | DO statement WHILE '(' expression ')' ';'
                {
                    $$ = new ASTloop_statement (oslcompiler,
                                                ASTloop_statement::LoopDo,
                                                NULL, $5, NULL, $2);
                    $$->sourceline (@1.first_line);
                }
        | FOR '(' 
                {
                    oslcompiler->symtab().push (); // new declaration scope
                }
          for_init_statement expression_opt ';' expression_opt ')' statement
                {
                    $$ = new ASTloop_statement (oslcompiler,
                                                ASTloop_statement::LoopFor,
                                                $4, $5, $7, $9);
                    $$->sourceline (@1.first_line);
                    oslcompiler->symtab().pop ();
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
        | string_literal_group  { $$ = new ASTliteral (oslcompiler, ustring($1)); }
        | variable_ref
        | incdec_op variable_lvalue 
                {
                    $$ = new ASTpreincdec (oslcompiler, $1, $2);
                }
        | binary_expression
        | unary_op expression %prec UMINUS_PREC
                {
                    // Correct for literal +float -float
                    if ($1 == ASTNode::Add) {
                        $$ = $2;
                    } else if ($1 == ASTNode::Sub &&
                               $2->nodetype() == ASTNode::literal_node &&
                               $2->typespec().is_numeric()) {
                         ((ASTliteral *)$2)->negate ();
                         $$ = $2;
                    } else {
                        $$ = new ASTunary_expression (oslcompiler, $1, $2);
                    }
                }
        | '(' expression ')'                    { $$ = $2; }
        | function_call
        | assign_expression
        | ternary_expression
        | typecast_expression
        | type_constructor
        ;

variable_lvalue
        : id_or_field
        | id_or_field '[' expression ']'
                {
                    $$ = new ASTindex (oslcompiler, $1, $3);
                }
        | id_or_field '[' expression ']' '[' expression ']'
                {
                    $$ = new ASTindex (oslcompiler, $1, $3, $6);
                }
        | id_or_field '[' expression ']' '[' expression ']' '[' expression ']'
                {
                    $$ = new ASTindex (oslcompiler, $1, $3, $6, $9);
                }
        ;

id_or_field
        : IDENTIFIER 
                {
                    $$ = new ASTvariable_ref (oslcompiler, ustring($1));
                }
        | variable_lvalue '.' IDENTIFIER
                {
                    $$ = new ASTstructselect (oslcompiler, $1, ustring($3));
                }
        ;

variable_ref
        : variable_lvalue incdec_op_opt
                {
                    if ($2)
                        $$ = new ASTpostincdec (oslcompiler, $2, $1);
                    else
                        $$ = $1;
                }
        ;

binary_expression
        : expression OR_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Or, $1, $3);
                }
        | expression AND_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::And, $1, $3);
                }
        | expression '|' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::BitOr, $1, $3);
                }
        | expression '^' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Xor, $1, $3);
                }
        | expression '&' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::BitAnd, $1, $3);
                }
        | expression EQ_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Equal, $1, $3);
                }
        | expression NE_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::NotEqual, $1, $3);
                }
        | expression '>' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Greater, $1, $3);
                }
        | expression GE_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::GreaterEqual, $1, $3);
                }
        | expression '<' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Less, $1, $3);
                }
        | expression LE_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::LessEqual, $1, $3);
                }
        | expression SHL_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::ShiftLeft, $1, $3);
                }
        | expression SHR_OP expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::ShiftRight, $1, $3);
                }
        | expression '+' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Add, $1, $3);
                }
        | expression '-' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Sub, $1, $3);
                }
        | expression '*' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Mul, $1, $3);
                }
        | expression '/' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Div, $1, $3);
                }
        | expression '%' expression
                {
                    $$ = new ASTbinary_expression (oslcompiler, 
                                    ASTNode::Mod, $1, $3);
                }
        ;

unary_op
        : '-'                           { $$ = ASTNode::Sub; }
        | '+'                           { $$ = ASTNode::Add; }
        | '!'                           { $$ = ASTNode::Not; }
        | '~'                           { $$ = ASTNode::Compl; }
        ;

incdec_op_opt
        : incdec_op
        | /* empty */                   { $$ = 0; }
        ;

incdec_op
        : INCREMENT                     { $$ = ASTNode::Incr; }
        | DECREMENT                     { $$ = ASTNode::Decr; }
        ;

type_constructor
        : simple_typename '(' expression_list ')'
                {
                    $$ = new ASTtype_constructor (oslcompiler,
                                                  TypeSpec (lextype ($1)), $3);
                }
        ;

function_call
        : IDENTIFIER '(' function_args_opt ')'
                {
                    $$ = new ASTfunction_call (oslcompiler, ustring($1), $3);
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
                                ASTNode::Assign, $3); }
        | variable_lvalue MUL_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
                	        ASTNode::Mul, $3); }
        | variable_lvalue DIV_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::Div, $3); }
        | variable_lvalue ADD_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::Add, $3); }
        | variable_lvalue SUB_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::Sub, $3); }
        | variable_lvalue BIT_AND_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::BitAnd, $3); }
        | variable_lvalue BIT_OR_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::BitOr, $3); }
        | variable_lvalue XOR_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::Xor, $3); }
        | variable_lvalue SHL_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::ShiftLeft, $3); }
        | variable_lvalue SHR_ASSIGN expression
                { $$ = new ASTassign_expression (oslcompiler, $1,
				ASTNode::ShiftRight, $3); }
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

string_literal_group
        : STRING_LITERAL
        | string_literal_group STRING_LITERAL
                {
                    $$ = ustring (std::string($1) + std::string($2)).c_str();
                }
        ;

%%



void
yyerror (const char *err)
{
    oslcompiler->error (oslcompiler->filename(), oslcompiler->lineno(),
                        "Syntax error: %s", err);
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
