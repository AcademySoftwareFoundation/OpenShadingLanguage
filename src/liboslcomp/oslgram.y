/* Copyright Contributors to the Open Shading Language project.
 * SPDX-License-Identifier: BSD-3-Clause
 * https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
 */

/** Parser for Open Shading Language
 **/

%define api.pure full
%lex-param   { void *scanner }
%lex-param   { OSL::pvt::OSLCompilerImpl *oslcompiler }
%parse-param { void *scanner }
%parse-param { OSL::pvt::OSLCompilerImpl *oslcompiler }

%{

// C++ declarations

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <string>

#include "oslcomp_pvt.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wparentheses-equality"
#endif

#if (OSL_CLANG_VERSION >= 150000) || (OSL_APPLE_CLANG_VERSION >= 140000) \
    || (OSL_INTEL_CLANG_VERSION >= 140000)
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif

using namespace OSL;
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

%{
OSL_NAMESPACE_ENTER
namespace pvt {

int yylex(YYSTYPE* yylval_param, YYLTYPE* yylloc_param, void* yyscanner, OSLCompilerImpl* oslcompiler);
void yyerror (YYLTYPE* yylloc_param, void* yyscanner, OSLCompilerImpl* oslcompiler, const char* err);

TypeDesc osllextype (int lex);

} // namespace pvt
OSL_NAMESPACE_EXIT
%}


// Tell Bison to track locations for improved error messages
%locations


// Define the terminal symbols.
%token <s> IDENTIFIER STRING_LITERAL
%token <i> INT_LITERAL
%token <f> FLOAT_LITERAL
%token <i> COLORTYPE FLOATTYPE INTTYPE MATRIXTYPE 
%token <i> NORMALTYPE POINTTYPE STRINGTYPE VECTORTYPE VOIDTYPE
%token <i> CLOSURE OUTPUT PUBLIC STRUCT
%token <i> BREAK CONTINUE DO ELSE FOR IF_TOKEN ILLUMINATE ILLUMINANCE RETURN WHILE
%token <i> RESERVED


// Define the nonterminals 
%type <n> shader_file 
%type <n> global_declarations_opt global_declarations global_declaration
%type <n> shader_or_function_declaration
%type <n> formal_params_opt formal_params formal_param
%type <n> metadata_block_opt metadata metadatum
%type <n> function_declaration
%type <n> function_body_or_just_decl
%type <n> struct_declaration
%type <i> field_declarations field_declaration
%type <n> typed_field_list typed_field
%type <n> variable_declaration def_expressions def_expression
%type <n> initializer_opt initializer initializer_list_opt initializer_list 
%type <n> compound_initializer init_expression_list init_expression
%type <n> init_expression_list_rev
%type <i> outputspec arrayspec simple_typename
%type <i> typespec typespec_or_shadertype
%type <n> statement_list statement scoped_statements local_declaration
%type <n> conditional_statement loop_statement loopmod_statement
%type <n> return_statement
%type <n> for_init_statement
%type <n> expression compound_expression
%type <n> expression_list expression_opt compound_expression_opt
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
%right <i> UMINUS_PREC NOT_OP '~'
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
        : shader_or_function_declaration    { $$ = 0; }
        | struct_declaration                { $$ = 0; }
        ;

shader_or_function_declaration
        : typespec_or_shadertype IDENTIFIER
                {
                    if ($1 == (int)ShaderType::Unknown) {
                        // It's a function declaration, not a shader
                        oslcompiler->symtab().push ();  // new scope
                        oslcompiler->typespec_stack().push (oslcompiler->current_typespec());
                    }
                }
          metadata_block_opt '(' 
                {
                    if ($1 != (int)ShaderType::Unknown)
                        oslcompiler->declaring_shader_formals (true);
                }
          formal_params_opt ')'
                {
                    oslcompiler->declaring_shader_formals (false);
                }
          metadata_block_opt function_body_or_just_decl
                {
                    if ($1 == (int)ShaderType::Unknown) {
                        // Function declaration
                        oslcompiler->symtab().pop ();  // restore scope
                        ASTfunction_declaration *f;
                        f = new ASTfunction_declaration (oslcompiler,
                                                         oslcompiler->typespec_stack().top(),
                                                         ustring($2), $7 /*formals*/,
                                                         $11 /*statements*/,
                                                         NULL /*metadata*/,
                                                         @2.first_line);
                        oslcompiler->remember_function_decl (f);
                        f->add_meta (concat($4, $10));
                        $$ = f;
                        oslcompiler->typespec_stack().pop ();
                    } else {
                        // Shader declaration
                        $$ = new ASTshader_declaration (oslcompiler, $1,
                                                        ustring($2), $7 /*formals*/,
                                                        $11 /*statements*/,
                                                        concat($4,$10) /*meta*/);
                        $$->sourceline (@2.first_line);
                        if (oslcompiler->shader_is_defined()) {
                        yyerror (&yylloc, scanner, oslcompiler, YY_("Only one shader is allowed per file."));
                            delete $$;
                            $$ = NULL;
                        } else {
                            oslcompiler->shader ($$);
                        }
                    }
                }
        ;

formal_params_opt
        : formal_params
        | /* empty */                   { $$ = 0; }
        ;

formal_params
        : formal_param
        | formal_params ',' formal_param        { $$ = concat ($1, $3); }
        | formal_params ','                     { $$ = $1; }
        ;

formal_param
        : outputspec typespec IDENTIFIER initializer_opt metadata_block_opt
                {
                    TypeSpec t = oslcompiler->current_typespec();
                    bool is_output = $1;
                    auto var = new ASTvariable_declaration (oslcompiler,
                                            t, ustring($3), $4 /*init*/,
                                            oslcompiler->declaring_shader_formals() /*isparam*/,
                                            false /*ismeta*/, is_output,
                                            false /*instlist*/, @3.first_line);
                    if (! oslcompiler->declaring_shader_formals() && !is_output) {
                        // these are function formals, not shader formals,
                        var->sym()->readonly (true);
                    }
                    var->add_meta ($5);
                    $$ = var;
                }
        | outputspec typespec IDENTIFIER arrayspec initializer_list_opt metadata_block_opt
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    t.make_array ($4);
                    auto var = new ASTvariable_declaration (oslcompiler, t, 
                                            ustring($3), $5 /*init*/,
                                            oslcompiler->declaring_shader_formals() /*isparam*/,
                                            false /*ismeta*/, $1 /*isoutput*/,
                                            true /* initlist */, @3.first_line);
                    var->add_meta ($6);
                    $$ = var;
                }
        | outputspec typespec IDENTIFIER initializer_list_opt metadata_block_opt
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    if (! t.is_structure() && ! t.is_triple() && ! t.is_matrix())
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Can't use '= {{...}}' initializer "
                                              "except with arrays, structs, vectors, "
                                              "or matrix ({})", $3);
                    auto var = new ASTvariable_declaration (oslcompiler, t,
                                            ustring($3), $4 /*init*/,
                                            oslcompiler->declaring_shader_formals() /*isparam*/,
                                            false /*ismeta*/, $1 /*isoutput*/,
                                            true /* initializer list */,
                                            @3.first_line);
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
        | metadata ','                  
                { 
                    $$ = $1; 
                }
        ;

metadatum
        : simple_typename IDENTIFIER initializer
                {
                    auto var = new ASTvariable_declaration (oslcompiler, osllextype($1),
                                           ustring ($2), $3, false /* isparam */,
                                           true /* ismeta */,
                                           false /* isoutput */,
                                           false /* initlist */,
                                           @2.first_line);
                    $$ = var;
                }
        | simple_typename IDENTIFIER arrayspec initializer_list
                {
                    TypeDesc simple = osllextype ($1);
                    simple.arraylen = $3;
                    if (simple.arraylen < 1)
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Invalid array length for {}", $2);
                    TypeSpec t (simple, false);
                    auto var = new ASTvariable_declaration (oslcompiler, t, 
                                     ustring ($2), $4, false,
                                     true /* ismeata */, false /* output */,
                                     true /* initializer list */,
                                     @2.first_line);
                    $$ = var;
                }
        ;

function_body_or_just_decl
        : '{' statement_list '}'        { $$ = $2; }
        | ';'                           { $$ = NULL; }
        ;

function_declaration
        : typespec IDENTIFIER 
                {
                    oslcompiler->symtab().push ();  // new scope
                    oslcompiler->typespec_stack().push (oslcompiler->current_typespec());
                }
          '(' formal_params_opt ')' metadata_block_opt function_body_or_just_decl
                {
                    oslcompiler->symtab().pop ();  // restore scope
                    auto f = new ASTfunction_declaration (oslcompiler,
                                                     oslcompiler->typespec_stack().top(),
                                                     ustring($2), $5, $8, NULL,
                                                     @2.first_line);
                    oslcompiler->remember_function_decl (f);
                    f->add_meta ($7);
                    $$ = f;
                    oslcompiler->typespec_stack().pop ();
                }
        ;

struct_declaration
        : STRUCT IDENTIFIER '{' 
                {
                    ustring name ($2);
                    Symbol *s = oslcompiler->symtab().clash (name);
                    if (s) {
                        oslcompiler->errorfmt(oslcompiler->filename(), oslcompiler->lineno(),
                                              "\"{}\" already declared in this scope", name);
                        // FIXME -- print the file and line of the other definition
                    }
                    if (OIIO::Strutil::starts_with (name, "___")) {
                        oslcompiler->errorfmt(oslcompiler->filename(), oslcompiler->lineno(),
                                              "\"{}\" : sorry, can't start with three underscores", name);
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
        | field_declarations field_declaration 
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
                    ustring name ($1);
                    TypeSpec t = oslcompiler->current_typespec();
                    StructSpec *s = oslcompiler->symtab().current_struct();
                    if (s->lookup_field (name) >= 0)
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Field \"{}\" already exists in struct \"{}\"",
                                              name, s->name());
                    else
                        oslcompiler->symtab().add_struct_field (t, name);
                    $$ = 0;
                }
        | IDENTIFIER arrayspec
                {
                    // Grab the current declaration type, modify it to be array
                    ustring name ($1);
                    TypeSpec t = oslcompiler->current_typespec();
                    t.make_array ($2);
                    if (t.arraylength() < 1)
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Invalid array length for {}", name);
                    StructSpec *s = oslcompiler->symtab().current_struct();
                    if (s->lookup_field (name) >= 0)
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Field \"{}\" already exists in struct \"{}\"",
                                              name, s->name());
                    else
                        oslcompiler->symtab().add_struct_field (t, name);
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
        | def_expressions ',' def_expression    { $$ = concat ($1, $3); }
        ;

def_expression
        : IDENTIFIER initializer_opt
                {
                    TypeSpec t = oslcompiler->current_typespec();
                    $$ = new ASTvariable_declaration (oslcompiler,
                                      t, ustring($1), $2, false /*isparam*/,
                                      false /*ismeta*/, false /*isoutput*/,
                                      false /*initlist*/, @1.first_line);
                }
        | IDENTIFIER arrayspec initializer_list_opt
                {
                    // Grab the current declaration type, modify it to be array
                    TypeSpec t = oslcompiler->current_typespec();
                    t.make_array ($2);
                    if ($2 < 1)
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Invalid array length for {}", $1);
                    $$ = new ASTvariable_declaration (oslcompiler, t,
                                 ustring($1), $3, false, false, false,
                                 true /* initlist */, @1.first_line);
                }
        | IDENTIFIER initializer_list
                {
                    TypeSpec t = oslcompiler->current_typespec();
                    if (! t.is_structure() && ! t.is_triple() && ! t.is_matrix())
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Can't use '= {{...}}' initializer "
                                              "except with arrays, struct, vectors, "
                                              "or matrix ({})", $1);
                    $$ = new ASTvariable_declaration (oslcompiler, t,
                                 ustring($1), $2, false, false, false,
                                 true /* initlist */, @1.first_line);
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
        : '=' compound_initializer      { $$ = $2; }
        ;

compound_initializer
        : '{' init_expression_list '}'
                {
                    $$ = new ASTcompound_initializer (oslcompiler, $2);
                    $$->sourceline (@1.first_line);
                }
        | '{' '}'
                {
                    if (!oslcompiler->declaring_shader_formals())
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Empty compound initializers '{{ }}' "
                                              "only allowed for shader parameters.");
                    $$ = new ASTcompound_initializer (oslcompiler, nullptr);
                    $$->sourceline (@1.first_line);
                }
        ;

init_expression_list
        : init_expression
        | init_expression_list_rev ',' init_expression
                {
                #ifdef OIIO_REFCNT_HAS_RELEASE
                    // Left recursion is much more efficient for Bison, but
                    // concatenating the sole last expression ($3) on the
                    // far end of the running list ($1) leads to an
                    // inadvertently quadratic algorithm, which is very
                    // painful for long initializer lists of thousands of
                    // items (like for a big table). So we use a trick:
                    // accumulate the list in referse order (lets us prepend
                    // as O(1) instead of append with O(n)) and then reverse
                    // it right before we return the list. BUT... note that
                    // we only do this for sufficiently new OIIO that lets
                    // us safely convert the ref-counted pointer to a raw
                    // pointer.
                    ASTNode::ref revlist = concat ($3, $1);
                    revlist = reverse (revlist);
                    $$ = revlist.release ();
                #else
                    // OIIO too old to support reverse(), so just do it
                    // the old way, normal order.
                    $$ = concat ($1, $3);
                #endif
                }
        ;

init_expression_list_rev
        : init_expression
        | init_expression_list_rev ',' init_expression
                {
                #ifdef OIIO_REFCNT_HAS_RELEASE
                    $$ = concat ($3, $1);
                    // NOTE: intentionally concat in reverse order!
                    // Because this is init_expression_list_rev!
                #else
                    // OIIO too old to support reverse(), so just do it
                    // the old way, normal order.
                    $$ = concat ($1, $3);
                #endif
                }
        ;

init_expression
        : expression
        | compound_initializer
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
        : '[' INT_LITERAL ']'
                {
                    if ($2 < 1)
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Invalid array length ({})", $2);
                    $$ = $2;
                }
        | '[' ']'                       { $$ = -1; }
        ;

/* typespec operates by merely setting the current_typespec */
typespec
        : simple_typename
                {
                    oslcompiler->current_typespec (TypeSpec (osllextype ($1)));
                    $$ = 0;
                }
        | CLOSURE simple_typename
                {
                    oslcompiler->current_typespec (TypeSpec (osllextype ($2), true));
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
                        oslcompiler->errorfmt(oslcompiler->filename(),
                                              oslcompiler->lineno(),
                                              "Unknown struct name: {}", $1);
                    }
                    $$ = 0;
                }
        ;

/* either a typespec or a shader type is allowable here */
typespec_or_shadertype
        : simple_typename
                {
                    oslcompiler->current_typespec (TypeSpec (osllextype ($1)));
                    $$ = 0;
                }
        | CLOSURE simple_typename
                {
                    oslcompiler->current_typespec (TypeSpec (osllextype ($2), true));
                    $$ = 0;
                }
        | IDENTIFIER /* struct name or shader type name */
                {
                    /* N.B. Shader types are considered obsolete. We are
                     * promoting 'shader' for all, now (OSL 2.0). Some day
                     * we may add a warning for using the old names.
                     */
                    ustring name ($1);
                    if (name == "shader")
                        $$ = (int)ShaderType::Generic;
                    else if (name == "surface")
                        $$ = (int)ShaderType::Surface;
                    else if (name == "displacement")
                        $$ = (int)ShaderType::Displacement;
                    else if (name == "volume")
                        $$ = (int)ShaderType::Volume;
                    else {
                        Symbol *s = oslcompiler->symtab().find (name);
                        if (s && s->is_structure())
                            oslcompiler->current_typespec (TypeSpec ("", s->typespec().structure()));
                        else {
                            oslcompiler->current_typespec (TypeSpec (TypeDesc::UNKNOWN));
                            oslcompiler->errorfmt(oslcompiler->filename(),
                                                  oslcompiler->lineno(),
                                                  "Unknown struct name: {}", $1);
                        }
                        $$ = (int)ShaderType::Unknown;
                    }
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
        | compound_expression ';'
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
        : IF_TOKEN '(' compound_expression ')' statement
                {
                    $$ = new ASTconditional_statement (oslcompiler, $3, $5);
                    $$->sourceline (@1.first_line);
                }
        | IF_TOKEN '(' compound_expression ')' statement ELSE statement
                {
                    $$ = new ASTconditional_statement (oslcompiler, $3, $5, $7);
                    $$->sourceline (@1.first_line);
                }
        ;

loop_statement
        : WHILE '(' compound_expression ')' statement
                {
                    $$ = new ASTloop_statement (oslcompiler,
                                                ASTloop_statement::LoopWhile,
                                                NULL, $3, NULL, $5);
                    $$->sourceline (@1.first_line);
                }
        | DO statement WHILE '(' compound_expression ')' ';'
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
          for_init_statement compound_expression_opt ';' compound_expression_opt ')' statement
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
        | RETURN compound_initializer ';'
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
        | expression_list ',' expression        { $$ = concat ($1, $3); }
        ;

expression_opt
        : expression
        | /* empty */                   { $$ = 0; }
        ;

compound_expression_opt
        : compound_expression
        | /* empty */                   { $$ = 0; }
        ;

compound_expression
        : expression
        | expression ',' compound_expression
                {
                    $$ = new ASTcomma_operator (oslcompiler, concat ($1, $3));
                }

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
                        $$->sourceline (@1.first_line);
                    }
                }
        | '(' compound_expression ')'
                {
                    if ($2->nodetype() == ASTNode::comma_operator_node) {
                        // Warning for comma operator ïn parenthesized
                        // expressions, which sometimes happens when somebody
                        // forgets the proper syntax for triple constructors:
                        //     color x = Cd * (a, b, c); // same as:  x = Cd * c
                        // when they really meant
                        //     color x = Cd * color(a, b, c);
                        oslcompiler->warningfmt(oslcompiler->filename(),
                                                @1.first_line,
                                                "Comma operator inside parenthesis is probably an error -- it is not a vector/color.");
                    }
                    $$ = $2;
                }
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
                    $$->sourceline (@2.first_line);
                }
        | id_or_field '[' expression ']' '[' expression ']'
                {
                    $$ = new ASTindex (oslcompiler, $1, $3, $6);
                    $$->sourceline (@2.first_line);
                }
        | id_or_field '[' expression ']' '[' expression ']' '[' expression ']'
                {
                    $$ = new ASTindex (oslcompiler, $1, $3, $6, $9);
                    $$->sourceline (@2.first_line);
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
                    $$->sourceline (@2.first_line);
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
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Or, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression AND_OP expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::And, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '|' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::BitOr, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '^' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Xor, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '&' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::BitAnd, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression EQ_OP expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Equal, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression NE_OP expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::NotEqual, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '>' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Greater, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression GE_OP expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::GreaterEqual, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '<' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Less, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression LE_OP expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::LessEqual, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression SHL_OP expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::ShiftLeft, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression SHR_OP expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::ShiftRight, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '+' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Add, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '-' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Sub, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '*' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Mul, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '/' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Div, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        | expression '%' expression
                {
                    $$ = ASTbinary_expression::make(oslcompiler,
                                    ASTNode::Mod, $1, $3);
                    $$->sourceline (@2.first_line);
                }
        ;

unary_op
        : '-'                           { $$ = ASTNode::Sub; }
        | '+'                           { $$ = ASTNode::Add; }
        | '!'                           { $$ = ASTNode::Not; }
        | NOT_OP                        { $$ = ASTNode::Not; }
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
                                                  TypeSpec (osllextype ($1)), $3);
                    $$->sourceline (@1.first_line);
                }
        ;

function_call
        : IDENTIFIER '(' function_args_opt ')'
                {
                    $$ = new ASTfunction_call (oslcompiler, ustring($1), $3);
                    $$->sourceline (@1.first_line);
                }
        ;

function_args_opt
        : function_args
        | /* empty */                   { $$ = 0; }
        ;

function_args
        : expression
        | compound_initializer
        | function_args ',' expression     { $$ = concat ($1, $3); }
        | function_args ',' compound_initializer     { $$ = concat ($1, $3); }
        ;

assign_expression
        : variable_lvalue '=' expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::Assign, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue MUL_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::Mul, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue DIV_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::Div, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue ADD_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::Add, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue SUB_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::Sub, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue BIT_AND_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::BitAnd, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue BIT_OR_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::BitOr, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue XOR_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::Xor, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue SHL_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::ShiftLeft, $3);
                    $$->sourceline (@2.first_line);
                }
        | variable_lvalue SHR_ASSIGN expression
                {
                    $$ = new ASTassign_expression (oslcompiler, $1, ASTNode::ShiftRight, $3);
                    $$->sourceline (@2.first_line);
                }
        ;

ternary_expression
        : expression '?' expression ':' expression
                {
                    $$ = new ASTternary_expression (oslcompiler, $1, $3, $5);
                    $$->sourceline (@2.first_line);
                }
        ;

typecast_expression
        : '(' simple_typename ')' expression
                {
                    $$ = new ASTtypecast_expression (oslcompiler, 
                                                     TypeSpec (osllextype ($2)),
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
OSL::pvt::yyerror (YYLTYPE* yylloc_param, void* yyscanner, OSLCompilerImpl* oslcompiler, const char* err)
{
    oslcompiler->errorfmt(oslcompiler->filename(), oslcompiler->lineno(),
                          "Syntax error: {}", err);
}



// Convert from the lexer's symbolic type (COLORTYPE, etc.) to a TypeDesc.
inline TypeDesc
OSL::pvt::osllextype (int lex)
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
