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


(chap-syntax)=
# Language Syntax

The body of a shader is a sequence of individual *statements*.
This chapter describes the types of statements and control-flow patterns
in OSL.

Statements in OSL include the following types of constructs:

* Scoped statements.
* Variable declarations.
* Expressions.
* Assignments.
* Control flow: `if`, `else`, `while`, `do`, `for`, `break`, `continue`
* Function declarations.

#### Scoping

Any place where it is legal to have a statement, it is legal to have multiple
statements enclosed by curly braces `{ }`.  This is called a *scope*.  Any
variables or functions declared within a scope are only visible within that
scope, and only may be used after their declaration.  Variables or functions
that are referenced will always resolve to the matching name in the innermost
scope relative to its use.  For example

```
    float a = 1;      // Call this the "outer" 'a'
    float b = 2;
    {
        float a = 3;  // Call this the "inner" 'a'
        float c = 1;
        b = a;        // b gets 3, because a is resolved to the inner scope
    }
    b += c;           // ERROR -- c was only in the inner scope

```



## Variable declarations and assignments

### Variable declarations

The syntax for declaring a variable in OSL is:

> *type* *name* 
>
> *type* *name* `=` *value*

where

* *type* is one of the basic data types, described earlier.
* *name* is the name of the variable you are declaring.
* If you wish to initialize your variable an initial value, you may
  immediately assign it a *value*, which may be any valid expression.

You may declare several variables of the same type in a single
declaration by separating multiple variable names by commas:

> *type* *name1* `,` *name2* ...
> 
> *type* *name1* [ = *value1*] `,`  *name2* [ `=` *value2* ] ...


Some examples of variable declarations are

```
    float a;           // Declare; current value is undefined
    float b = 1;       // Declare and assign a constant initializer
    float c = a*b;     // Computed initializer
    float d, e = 2, f; // Declare several variables of the same type
```

### Arrays

Arrays are also supported, declared as follows:

> *type variablename* `[` *arraylen* `}`
>
> *type variablename* `[` *arraylen* `}` `=` `{` *init0 `,` *init1* ... `}`

Array variables in OSL must have a constant length (though function parameters
and shader parameters may have undetermined length).  Some examples of array
variable declarations are:

```
    float d[10];                       // Declare an uninitialized array
    float c[3] = { 0.1, 0.2, 3.14 };   // Initialize the array
```

### Structures

Structures are used to group several fields of potentially different
types into a single object that can be referred to by name.  The syntax
for declaring a structure type is:

\spc `struct` *structname* {\cf \{} 

\spc\spc *type1* *fieldname1* `;`

\spc\spc ...

\spc\spc *typeN* *fieldnameN* `;`

\spc `) ;`

You may then use the structure type name to declare structure variables
as you would for any of the built-in types:

\spc *structname* *variablename* `;`

\spc *structname* *variablename* {\cf = \{ }
*initializer1* `,` ... *initializerN* `) ;`

If initializers are supplied, each field of the structure will be
initialized with the initializer in the corresponding position, which
is expected to be of the appropriate type.

Structure elements are accessed in the same way as other C-like
languages, using the `dot' operator: 

\spc *variablename*{\cf .} *fieldname*

Examples of declaration and use of structures:

```
    struct ray {
        point pos;
        vector dir;
    };

    ray r;   // Declare a structure
    ray s = { point(0,0,0), vector(0,0,1) };  // declare and initialize
    r.pos = point (1, 0, 0);  // Assign to one field
```

It is permitted to have a structure field that is an array, as well as
to have an array of structures.  But it is not permitted for one
structure to have a field that is another structure.

Please refer to Section~\ref{sec:types:struct} for more information
on using `struct`.


(sec-expressions)=
## Expressions

The expressions available in OSL include the following:

* Constants: integer (e.g., `1`, `42`), floating-point (e.g. `1.0`, `3`,
  `-2.35e4`), or string literals (e.g., `"hello"`)

* point, vector, normal, or matrix constructors, for example:

  ```
      color (1, 0.75, 0.5)
      point ("object", 1, 2, 3)
  ```

  If all the arguments to a constructor are themselves constants, the
  constructed point is treated like a constant and has no runtime cost.
  That is, `color(1,2,3)` is treated as a single constant entity, not
  assembled bit by bit at runtime.

* Variable or parameter references

* An individual element of an array (using `[ ]`)

* An individual component of a `color`, `point`, `vector`, `normal` (using
  `[ ]`), or of a `matrix` (using `[][]`)

* prefix and postfix increment and decrement operators:

  Operator | Meaning
  :-- | :--
  *varref* `++` | post-increment
  *varref* `--` | post-decrement
  `++` *varref* | pre-increment
  `--` *varref* | pre-decrement

  The post-increment and post-decrement (e.g., {\cf a++}) returns the old
  value, then increments or decrements the variable; the pre-increment and
  pre-decrement ({\cf ++a}) will first increment or decrement the
  variable, then return the new value.

* Unary and binary arithmetic operators on other expressions:

  Operator | Meaning
  :-- | :--
  `-` *expr*         | negation
  `~` *expr*         | bitwise complement
  *expr* `*`  *expr* | multiplication
  *expr* `/`  *expr* | division
  *expr* `+`  *expr* | addition
  *expr* `-`  *expr* | subtraction
  *expr* `%`  *expr* | integer modulus
  *expr* `<<` *expr* | integer shift left
  *expr* `>>` *expr* | integer shift right
  *expr* `&`  *expr* | bitwise and
  *expr* `\|` *expr* | bitwise or
  *expr* `^`  *expr* | bitwise exclusive or
  
  The operators `+`, `-`, `*`, `/`, and the unary `-` (negation) may be used
  on most of the numeric types.  For multicomponent types (`color`, `point`,
  `vector`, `normal`, `matrix`), these operators combine their arguments on a
  component-by-component basis. The only operators that may be applied to the
  `matrix` type are `*` and `/`, which respectively denote matrix-matrix
  multiplication and matrix multiplication by the inverse of another matrix.
  
  The integer and bit-wise operators `%`, `<<`, `>>`, `&`, `|`, `^`, and `~`
  may only be used with expressions of type `int`.
  
  For details on which operators are allowed, please consult the operator
  tables for each individual type in Chapter [](#chap-types).

* Relational operators (all lower precedence than the arithmetic operators):

  Operator | Meaning
  :-- | :--
  *expr* `==` *expr* | equal to
  *expr* `!=` *expr* | not equal to
  *expr* `<`  *expr* | less then
  *expr* `<=` *expr* | less than or equal to
  *expr* `>`  *expr* | greater than
  *expr* `>=` *expr* | greater than or equal
  
  The `==` and `!=` operators may be performed between any two values of equal
  type, and are performed component-by-component for multi-component types.
  The `<`, `<=`, `>`, `>=` may not be used to compare multi-component types.
  
  An `int` expression may be compared to a `float` (and is treated as if
  they are both `float`).  A `float` expression may be compared to a
  multi-component type (and is treated as a multi-component type as if
  constructed from a single float).
  
  Relation comparisons produce Boolean (true/false) values.  These
  are implemented as `int` values, 0 if false and 1 if true.

* Logical unary and binary operators:

  > `!` *expr*
  > *expr1* `&&` *expr2*
  > *expr1* `\|\|` *expr2*
  
  Note that the `not`, `and`, and `or` keywords are synonyms for `!`, `&&`,
  and `||`, respectively.
  
  For the logical operators, numeric expressions (`int` or `float`) are
  considered *true* if nonzero, *false* if zero. Multi-component types (such
  as `color`) are considered *true* any component is nonzero, *false* all
  components are zero.  Strings are considered *true* if they are nonempty,
  *false* if they are the empty string (`""`).

* another expression enclosed in parentheses: `( )`. Parentheses may be used
  to guarantee associativity of operations.

* Type casts, specified either by having the type name in parentheses in front
  of the value to cast (C-style typecasts) or the type name called as a
  constructor (C++-style type constructors):

  ```
        (vector) P            /* cast a point to a vector */
        (point) f             /* cast a float to a point */
        (color) P             /* cast a point to a color! */

        vector (P)            /* Means the same thing */
        point (f)
        color (P)
  ```

  The three-component types (`color`, `point`, `vector`, `normal`) may be cast
  to other three-component types.  A `float` may be cast to any of the
  three-component types (by placing the float in all three components) or to a
  `matrix` (which makes a matrix with all diagonal components being the
  `float`). Obviously, there are some type casts that are not allowed because
  they make no sense, like casting a `point` to a `float`, or casting a
  `string` to a numerical type.

* function calls

* assignment expressions:

  same thing as `var = var OP expr` :
  
  Operator | Meaning
  :-- | :--
  *var* `=` *expr*  | assign
  *var* `+=` *expr* | add
  *var* `-=` *expr* | subtract
  *var* `*=` *expr* | multiply
  *var* `/=` *expr* | divide
  *int-var* `&=` *int-expr* | bitwise and
  *int-var* `\|=` *int-expr* | bitwise or
  *int-var* `^=` *int-expr* | bitwise exclusive or
  *int-var* `<<=` *int-expr* | integer shift left
  *int-var* `>>=` *int-expr* | integer shift right
  
  Note that the integer and bit-wise operators are only allowed with `int`
  variables and expressions.  In general, `var OP= expr` is allowed only if
  `var = var OP expr` is allowed, and means exactly the same thing.  Please
  consult the operator tables for each individual type in Chapter
  [](#chap-types).

* ternary operator, just like C: 

  > *condition* `?` *expr1* `:` *expr2*

  This expression takes on the value of *expr1* if *condition* is true
  (nonzero), or *expr2* if *condition* is false (zero).


Please refer to Chapter [](#chap-types), where the section describing each
data type describes the full complement of operators that may be used with the
type.  Operator precedence in OSL is identical to that of C.


## Control flow: `if`, `while`, `do`, `for`

Conditionals in OSL just like in C or C++:

\begin{tabbing}
\hspace{0.5in} \= \hspace{0.3in} \= \kill
\> {\cf if (} *condition* {\cf )} \\
\> \> *truestatement*  
\end{tabbing}

and

\begin{tabbing}
\hspace{0.5in} \= \hspace{0.3in} \= \kill
\> {\cf if (} *condition* {\cf )} \\
\> \> *truestatement*  \\
\> {\cf else} \\
\> \> *falsestatement*  
\end{tabbing}

The statements can also be entire blocks, surrounded by curly
braces.  For example,

```
    if (s > 0.5) {
        x = s;
        y = 1;
    } else {
        x = s+t;
    }
```

The *condition* may be any valid expression, including:

* The result of any comparison operator (such as `<`, `==`, etc.).
* Any numeric expression (`int`, `color`, `point`, `vector`, `normal`,
  `matrix`), which is considered "true" if nonzero and "false" if zero.
* Any string expression, which is considered "true" if it is a nonempty
  string, "false" if it is the empty string (`""`).
* A closure, which is considered "true" if it's empty (not assigned, or
  initialized with {\cf =0}), and "false" if anything else has been assigned
  to it.
* A logical combination of expressions using the operators `!` (not), `&&`
  (logical "and"), or `||` (logical "or"). Note that `&&` and `\|\|`
  *short circuit* as in C, i.e. `A && B` will only evaluate B if A
  is true, and `A || B` will only evaluate B if A is false.

Repeated execution of statements for as long as a condition is true is
possible with a `while` statement:

> `while (` *condition* `)` *statements*

Or the test may happen after the body of the loop, with a `do/while` loop:

> `do` *statement* `while (` *condition* `)`

Also, `for` loops are also allowed:

> `for (` *initialization-statement* `;` *condition* `;` *iteration-statement* `)` \
>     *body-statements*

As in C++, a `for` loop's initialization may contain variable declarations and
initializations, which are scoped locally to the `for` loop itself.  For
example,

```
      for (int i = 0;  i < 3;  ++i) {
          ...
      }
```

As with `if` statements, loop conditions may be relations or numerical
quantities (which are considered "true" if nonzero, "false" if zero), or
strings (considered "true" if nonempty, "false" if the empty string `""`.

Inside the body of a loop, the `break` statement terminates the loop
altogether, and the `continue` statement skip to the end of the body and
proceeds to the next iteration of the loop.

## Functions

### Function definitions

You may define functions much like in C or C++.

> *return-type* *function-name* `(` *optional-parameters* `)` \
> `{` \
> *statements* \
> `}`

Parameters to functions are similar to shader parameters, except that they do
not permit initializers.  A function call must pass values for all formal
parameters.  Function parameters in OSL are all *passed by reference*, and are
read-only within the body of the function unless they are also designated as
`output` (in the same manner as output shader parameters).

Like for shaders, statements inside functions may be actual executions
(assignments, function call, etc.), local variable declarations (visible
only from within the body of the function), or local function
declarations (callable only from within the body of the function).

The return type may be any simple data type, a `struct`, or a 
`closure`.  Functions may not return arrays.  The return type may be
`void`, indicating that the function does not return a value (and
should not contain a `return` statement).  A `return` statement
inside the body of the function will halt execution of the function at
that point, and designates the value that will be returned (if not a
`void` function).

Functions may be *overloaded*.  That is, multiple functions may be
defined to have the same name, as long as they have differently-typed
parameters, so that when the function is called the list of arguments can
select which version of the function is desired. When there are multiple
potential matches, function versions whose argument types match exactly are
favored, followed by those that match with type coercion (for example,
passing an `int` when the function expects a `float`, or passing a
`float` when the function expects a `color`), and finally by trying to
match the return value to the type of variable the result is assigned to.

(sec-syntax-functioncalls)=
### Function calls

Function calls are very similar to C and related programming languages:

> *functionname* `(` *arg1* `,` ... `,` *argn* `)`

If the function returns a value (not `void`), you may use its value as an
expression.  It is fine to completely ignore the value of even a non-`void`
function.

In OSL, all arguments are passed by reference.  This generally will not be
noticeably different from C-style "pass by value" semantics, except if you
pass the same variable as two separate arguments to a function that modifies
an argument's value.

Certain functions allow optional arguments to be passed.  
Optional arguments are key-value pairs with the key passed as an argument and
the associated value passed as the subsequent argument:

> *functionname* `(` *arg1* `,` ... `,` *argn*  `,` *"optionalkey"*  `,` *optionalvalue* `,` ... `)` 


### Operator overloading

OSL permits \emph{operator overloading}, which is the practice of providing a
function that will be called when you use an operator like `+` or `*`. This is
especially handy when you use `struct` to define mathematical types and wish
for the usual math operators to work with them. Here is a typical example,
which also shows the special naming convention that allows operator
overloading:

```
    struct vector4 {
        float x, y, z, w;
    };

    vector4 __operator__add__ (vector4 a, vector4 b) {
        return vector4 (a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
    }

    shader test ()
    {
        vector4 a = vector4 (.2, .3, .4, .5);
        vector4 b = vector4 (1, 2, 3, 4);

        vector4 c = a + b;   // Will call __operator__add__(vector4,vector4)
        printf ("a+b = %g %g %g %g\n", c.x, c.y, c.z, c.w);
    }
```

The full list of these special function names is as follows (in order of
decreasing operator precedence):


Operator    | Overload function name | notes
:---------- | :----------- | :------
`-` | `__operator__neg__` | unary negation
`~` | `__operator__compl__` | unary bitwise 
`!` | `__operator__not__` | unary boolean 'not'
`*` | `__operator__mul__` | 
`/` | `__operator__div__` | 
`\%` | `__operator__mod__` | 
`+` | `__operator__add__` | 
`-` | `__operator__sub__` | 
`<<` | `__operator__shl__` |
`>>` | `__operator__shr__` |
`<` | `__operator__lt__` | 
`<=` | `__operator__le__` | 
`>` | `__operator__gt__` | 
`>=` | `__operator__ge__` |
`==` | `__operator__eq__` |
`!=` | `__operator__ne__` |
`\&` | `__operator__bitand__` |
`^` | `__operator__xor__` |
`\|` | `__operator__bitor__` |


## Global variables

*Global variables* (sometimes called *graphics state variables*) contain the
basic information that the renderer knows about the point being shaded, such
as position, surface orientation, and default surface color.  You need not
declare these variables; they are simply available by default in your shader.
Global variables available in shaders are listed in the following table:

Variable    | Description
:---------- | :-----------
`point P`   | Position of the point you are shading.  In a displacement shader, changing this variable displaces the surface.
`vector I`  | The *incident* ray direction, pointing from the viewing position to the shading position `P`.
`normal N`  | The surface "Shading" "normal of the surface at `P`.  Changing `N` yields bump mapping.
`normal Ng` | The true surface normal at `P`.  This can differ from `N`; `N` can be overridden in various ways including bump mapping and user-provided vertex normals, but `Ng` is always the true surface geometric normal of the surface at `P`.
`float u, v` | The 2D parametric coordinates of \P (on the particular geometric primitive you are shading).
`vector dPdu, dPdv` | Partial derivatives $\partial P/\partial u$ and $\partial P/\partial v$ tangent to the surface at `P`. 
`point Ps`   | Position at which the light is being queried (currently only used for light attenuation shaders)
`float time` | Current shutter time for the point being shaded.
`float dtime` | The amount of time covered by this shading sample.
`vector dPdtime` | How the surface position \P is moving per unit time
`closure color Ci` | Incident radiance --- a closure representing the color of the light leaving the surface from `P` in the direction `-I`.


Accessibility of variables by shader type:

Variable    | surface | displacement | volume
:---------- | ----- | -- | --
`P`         | R     | RW | R
`I`         | R     |    | R
`N`         | RW    | RW |
`Ng`        | R     | R  |
`dPdu`      | R     | R  |
`dPdv`      | R     | R  |
`Ps`        |       |    | R
`u, v`      | R     | R  | R
`time`      | R     | R  | R
`dtime`     | R     | R  | R
`dPdtime`   | R     | R  | R
`Ci`        | RW    |    | RW

