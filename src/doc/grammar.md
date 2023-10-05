---
numbering:
  heading_1: true
  heading_2: true
  heading_3: true
---

<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: CC-BY-4.0
-->


(chap-grammar)=
# Formal Language Grammar

This section gives the complete syntax of OSL.  Syntactic structures that have
a name ending in `-opt` are optional.  Structures surrounded by curly braces
`{ }` may be repeated 0 or more times.  Text in `typewriter` face indicates
literal text.  The $\epsilon$ character is used to indicate that it is
acceptable for there to be nothing (empty, no token).

#### Lexical elements


\<digit\> ::= "`0`" | "`1`" | "`2`" | "`3`" | "`4`" | "`5`" | "`6`" | "`7`" | "`8`" | "`9`"

\<digit-sequence\> ::= \<digit\> { \<digit\> }

\<hexdigit\> ::= \<digit\> | "`a`" | "`A`" | "`b`" | "`B`" | "`c`" | "`C`" | "`d`" | "`D`" | "`e`" | "`E`" | "`f`" | "`F`"

\<hexdigit-sequence\> ::= \<hexdigit\> { \<hexdigit\> }

\<integer\> ::= \<sign\> \<digit-sequence\>
<br>$~~~~~~~~~~$ | \<sign\> "`0x`" \<hexdigit-sequence\>

\<floating-point\> ::= \<digit-sequence\> \<decimal-part-opt\> \<exponent-opt\>
<br>$~~~~~~~~~~$ | \<decimal-part\> \<exponent-opt\>

\<decimal-part\> ::= '`.`' { \<digit\> }

\<exponent\> ::= '`e`' \<sign\> \<digit-sequence\>

\<sign\> ::= '`-`' | '`+`' | $\epsilon$

\<number\> ::= \<integer\>
<br>$~~~~~~~~~~$ | \<floating-point\>

\<char-sequence\> ::= { \<any-char\> } 

\<stringliteral> ::= `"` \<char-sequence\> `"`

\<identifier\> ::= \<letter-or-underscore\> { \<letter-or-underscore-or-digit\> }


#### Overall structure

\<shader-file\> ::= { \<global-declaration\> }

\<global-declaration\> ::= \<function-declaration\> 
<br>$~~~~~~~~~~$ | \<struct-declaration\>
<br>$~~~~~~~~~~$ | \<shader-declaration\>

\<shader-declaration\> ::= <br>$~~~~~~~~~~~~$ $~~~~~~~~~~$
    \<shadertype\> \<identifier\> \<metadata-block-opt\> 
    "(" \<shader-formal-params-opt\> ")" "{" \<statement-list\> "}"

\<shadertype\> ::= "displacement" |  "shader" | "surface" | "volume"

\<shader-formal-params\> ::= \<shader-formal-param\> { "," \<shader-formal-param\> }

\<shader-formal-param\> ::= \<outputspec\> \<typespec\> \<identifier\>
                               \<initializer\> \<metadata-block-opt\>
<br>$~~~~~~~~~~$ | \<outputspec\> \<typespec\> \<identifier\> \<arrayspec\>
          \<initializer-list\> \<metadata-block-opt\>

\<metadata-block> ::= "[[" \<metadata\> { "," \<metadata\> } "]]"

\<metadata\> ::= \<simple-typespec\> \<identifier\> \<initializer\>



#### Declarations

\<function-declaration\> ::= <br>$~~~~~~~~~~~~$ \<typespec\> \<identifier\> 
      "(" \<function-formal-params-opt\> ")" "{" \<statement-list\> "}"

\<function-formal-params\> ::= \<function-formal-param\> { "," \<function-formal-param\> }

\<function-formal-param\> ::= \<outputspec\> \<typespec\> \<identifier\> \<arrayspec-opt\>

\<outputspec> ::= "output" | $\epsilon$

\<struct-declaration\> ::= "struct" \<identifier\> "{" \<field-declarations> "}" ";"

\<field-declarations\> ::= \<field-declaration\> { \<field-declaration\> }

\<field-declaration\> ::= \<typespec\> \<typed-field-list\> ";"

\<typed-field-list\> ::= \<typed-field\> { "," \<typed-field\> }

\<typed-field\> ::= \<identifier\> \<arrayspec-opt\>

\<local-declaration\> ::= \<function-declaration\>
<br>$~~~~~~~~~~$ | \<variable-declaration\>

\<arrayspec> ::= "[" \<integer\> "]"
<br>$~~~~~~~~~~$ | "[" "]"

\<variable-declaration\> ::= \<typespec\> \<def-expressions> ";"

\<def-expressions\> ::= \<def-expression\> { "," <def-expression\> }

\<def-expression\> ::= \<identifier\> \<initializer-opt\>
<br>$~~~~~~~~~~$ | \<identifier\> \<arrayspec\> \<initializer-list-opt\>

\<initializer\> ::= "=" \<expression\>

\<initializer-list\> ::= "=" \<compound-initializer\>

\<compound-initializer\> ::= "{" \<init-expression-list\> "}"

\<init-expression-list\> ::= \<init-expression\> { "," \<init-expression\> }

\<init-expression\> ::= \<expression\> | \<compound-initializer\>

\<typespec\> ::= \<simple-typename> 
<br>$~~~~~~~~~~$ | "closure" \<simple-typename\>
<br>$~~~~~~~~~~$ | \<identifier-structname\>

\<simple-typename\> ::= 
"color"
| "float"
| "matrix"
| "normal"
| "point"
| "string"
| "vector"
| "void"



#### Statements

\<statement-list\> ::= \<statement\> { \<statement\> }

\<statement\> ::= 
\<compound-expression-opt\> ";"
<br>$~~~~~~~~~~$ | \<scoped-statements\>
<br>$~~~~~~~~~~$ | \<local-declaration\>
<br>$~~~~~~~~~~$ | \<conditional-statement\>
<br>$~~~~~~~~~~$ | \<loop-statement\>
<br>$~~~~~~~~~~$ | \<loopmod-statement\>
<br>$~~~~~~~~~~$ | \<return-statement\>

\<scoped-statements> ::= "{" \<statement-list-opt\> "}"

\<conditional-statement\> ::= <br>$~~~~~~~~~~~~$ "if" "(" \<compound-expression\> ")" \<statement\>
<br>$~~~~~~~~~~$ | "if" "(" \<compound-expression\> ")" \<statement\> "else" \<statement\>

\<loop-statement\> ::= <br>$~~~~~~~~~~~~$ "while" "(" \<compound-expression\> ")" \<statement\>
<br>$~~~~~~~~~~$ | "do" \<statement\> "while" "(" \<compound-expression\> ")" ";"
<br>$~~~~~~~~~~$ | "for" "(" \<for-init-statement-opt\>  \<compound-expression-opt\> ";" 
                \<compound-expression-opt\> ")" \<statement\>

\<for-init-statement\> ::= <br>$~~~~~~~~~~~~$ \<expression-opt\> ";"
<br>$~~~~~~~~~~$ | \<variable-declaration\>

\<loopmod-statement\> ::= "break" ";"
<br>$~~~~~~~~~~$ | "continue" ";"


\<return-statement\> ::= "return" \<expression-opt\> ";"



#### Expressions

\<expression-list\> ::= \<expression\> { "," \<expression\> }

\<expression\> ::= \<number\>
<br>$~~~~~~~~~~$ | \<stringliteral\>
<br>$~~~~~~~~~~$ | \<type-constructor\>
<br>$~~~~~~~~~~$ | \<incdec-op\> \<variable-ref\>
<br>$~~~~~~~~~~$ | \<expression\> \<binary-op\> \<expression\>
<br>$~~~~~~~~~~$ | \<unary-op\> \<expression\>
<br>$~~~~~~~~~~$ | "(" \<compound-expression\> ")"
<br>$~~~~~~~~~~$ | \<function-call\>
<br>$~~~~~~~~~~$ | \<assign-expression\>
<br>$~~~~~~~~~~$ | \<ternary-expression\>
<br>$~~~~~~~~~~$ | \<typecast-expression\>
<br>$~~~~~~~~~~$ | \<variable-ref\>
<br>$~~~~~~~~~~$ | \<compound-initializer\>

\<compound-expression\> ::= \<expression\> { "," \<expression\> }

\<variable-lvalue\> ::= \<identifier\> \<array-deref-opt\> \<component-deref-opt\>
<br>$~~~~~~~~~~$ | \<variable_lvalue> "[" \<expression\> "]"
<br>$~~~~~~~~~~$ | \<variable_lvalue> "." \<identifier\>

\<variable-ref\> ::= \<identifier\> \<array-deref-opt\> 

\<array-deref> ::= "[" \<expression\> "]"

\<component-deref> ::= "[" \<expression\> "]"
<br>$~~~~~~~~~~$ | "." \<component-field\>

\<component-field\> ::= "x" | "y" | "z" | "r" | "g" | "b"

\<binary-op> ::= "*" | "/" | "\%"
<br>$~~~~~~~~~~$ | "+" | "-" 
<br>$~~~~~~~~~~$ | "<<" | ">>"
<br>$~~~~~~~~~~$ | "<" | "<=" | ">" | ">=" 
<br>$~~~~~~~~~~$ | "==" | "!=" 
<br>$~~~~~~~~~~$ | "&"
<br>$~~~~~~~~~~$ | "^"
<br>$~~~~~~~~~~$ | "|"
<br>$~~~~~~~~~~$ | "&&" | "and"
<br>$~~~~~~~~~~$ | "||" | "or"

\<unary-op> ::= "-" | "~" | "!" | "not"

\<incdec-op> ::= "++" | "--"

\<type-constructor\> ::= \<typespec> "(" \<expression-list\> ")"

\<function-call\> ::= \<identifier\> "(" \<function-args-opt\> ")"

\<function-args\> ::= \<expression\> { "," \<expression\> }

\<assign-expression\> ::= \<variable-lvalue\> \<assign-op\> \<expression\>

\<assign-op> ::= "=" | "*=" | "/=" | "+=" | "-=" | "&=" | "|=" | "^=" |
"<<=" | ">>="

\<ternary-expression\> ::= \<expression\> "?" \<expression\> ":" \<expression\>

\<typecast-expression\> ::= "(" \<simple-typename> ")" \<expression\>

