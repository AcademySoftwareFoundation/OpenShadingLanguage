# Lexical structure

## Characters

Source code for OSL consists of ASCII or UTF-8 characters.

The characters for space, tab, carriage return, and linefeed are
collectively referred to as *whitespace*.  Whitespace characters
delimit identifiers, keywords, or other symbols, but other than that
have no syntactic meaning.  Multiple whitespace characters in a row
are equivalent to a single whitespace character.

Source code may be split into multiple lines, separated by end-of-line
markers (carriage return and/or linefeed).  Lines may be of any length
and end-of-line markers carry no significant difference from other 
whitespace, except that they terminate `//` comments and delimit
preprocessor directives.

## Identifiers

*Identifiers* are the names of variables, parameters, functions, and shaders.
In OSL, identifiers consist of one or more characters.  The first character
may be a letter (`A`-`Z` or `a`-`z`) or underscore (`_`), and subsequent
characters may be letters, underscore, or numerals (`0`-`9`).  Examples of
valid and invalid identifiers are:

```
    opacity       // valid
    Long_name42   // valid - letters, underscores, numbers are ok
    _foo          // valid - ok to start with an underscore

    2smart        // invalid - starts with a numeral
    bigbuck$      // invalid - $ is an illegal character
```


## Comments

*Comments* are text that are for the human reader of programs, and
are ignored entirely by the OSL compiler.  Just like in C++, there
are two ways to designate comments in OSL:

1. Any text enclosed by `/*` and `*/` will be considered a comment, even if
   the comment spans several lines.

   ```
   /* this is a comment */

   /* this is also
      a comment, spanning
      several lines */
   ```

2. Any text following `//`, up to the end of the current line, will be
   considered a comment.

   ```
   // This is a comment
   a = 3;   // another comment
   ```


## Keywords and reserved words

There are two sets of names that you may not use as identifiers:
keywords and reserved words.

The following are *keywords* that have special meaning in OSL:

    and break closure color continue do else emit float for if illuminance
    illuminate int matrix normal not or output point public return string
    struct vector void while

The following are *reserved words* that currently have no special meaning in
OSL, but we reserve them for possible future use, or because they are
confusingly similar to keywords in related programming languages:

    bool case catch char class const delete default double 
    enum extern false friend
    goto inline long new operator private protected 
    short signed sizeof static 
    switch template this throw true try typedef 
    uniform union unsigned varying virtual volatile


## Preprocessor

Shader source code is passed through a standard C preprocessor as a
first step in parsing.  

Preprocessor directives are designated by a hash mark `#` as the first
character on a line, followed by a preprocessor directive name. Whitespace may
optionally appear between the hash and the directive name.

OSL compilers support the full complement of C/C++ preprocessing directives,
including:

```
#define
#undef
#if
#ifdef
#ifndef
#elif
#else
#endif
#include
#pragma error "message"
#pragma once
#pragma osl ...
#pragma warning "message"
```

Additionally, the following preprocessor symbols will already be
defined by the compiler:

|  |  |
| ------ | :------- |
| `OSL_VERSION_MAJOR` | Major version (e.g., 1)  |
| `OSL_VERSION_MINOR` | Minor version (e.g., 9)  |
| `OSL_VERSION_PATCH` | Patch version (e.g., 3)  |
| `OSL_VERSION` | Combined version number = `10000*major + 100*minor + patch` (e.g., 10903 for version 1.9.3)  |
