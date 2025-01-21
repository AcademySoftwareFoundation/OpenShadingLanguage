// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once
#define OSL_UNITTEST_H

// If OSL_UNITTEST_VERBOSE is defined to nonzero, then *passing* CHECK tests
// will echo a pass message. The default is for passing tests to be silent.
// You can override this (i.e. printing a message for each passed check) by
// defining OSL_UNITTEST_VERBOSE prior to including osl-unittest.h.
#ifndef OSL_UNITTEST_VERBOSE
#    define OSL_UNITTEST_VERBOSE 0
#endif

// If OSL_UNITTEST_EXIT_ON_FAILURE is defined to nonzero, then failing CHECK
// tests will exit the entire shader. This is the default behavior. You can
// override this (i.e., print the failure message but continue running the
// rest of the shader) by defining OSL_UNITTEST_EXIT_ON_FAILURE prior to
// including osl-unittest.h.
#ifndef OSL_UNITTEST_EXIT_ON_FAILURE
#    define OSL_UNITTEST_EXIT_ON_FAILURE 1
#endif



void failmsg(string file, int line, string xs, int x, string ys, int y)
{
    printf("\nFAIL: %s:%d: %s (%d) == %s (%d)\n\n", file, line, xs, x, ys, y);
}

void failmsg(string file, int line, string xs, float x, string ys, float y)
{
    printf("\nFAIL: %s:%d: %s (%g) == %s (%g)\n\n", file, line, xs, x, ys, y);
}

void failmsg(string file, int line, string xs, color x, string ys, color y)
{
    printf("\nFAIL: %s:%d: %s (%g) == %s (%g)\n\n", file, line, xs, x, ys, y);
}

void failmsg(string file, int line, string xs, vector x, string ys, vector y)
{
    printf("\nFAIL: %s:%d: %s (%g) == %s (%g)\n\n", file, line, xs, x, ys, y);
}

#ifdef COLOR2_H
void failmsg(string file, int line, string xs, color2 x, string ys, color2 y)
{
    printf("\nFAIL: %s:%d: %s (%g %g) == %s (%g %g)\n\n",
           file, line, xs, x.r, x.a, ys, y.r, y.a);
}
#endif

#ifdef COLOR4_H
void failmsg(string file, int line, string xs, color4 x, string ys, color4 y)
{
    printf("\nFAIL: %s:%d: %s (%g %g) == %s (%g %g)\n\n",
           file, line, xs, x.rgb, x.a, ys, y.rgb, y.a);
}
#endif

#ifdef VECTOR2_H
void failmsg(string file, int line, string xs, vector2 x, string ys, vector2 y)
{
    printf("\nFAIL: %s:%d: %s (%g %g) == %s (%g %g)\n\n",
           file, line, xs, x.x, x.y, ys, y.x, y.y);
}
#endif

#ifdef VECTOR4_H
void failmsg(string file, int line, string xs, vector4 x, string ys, vector4 y)
{
    printf("\nFAIL: %s:%d: %s (%g %g %g %g) == %s (%g %g %g %g)\n\n",
           file, line, xs, x.x, x.y, x.z, x.w, ys, y.x, y.y, y.z, y.w);
}
#endif



// Macros to test conditions in shaders.
//
// A success by default will just silently move on, but if the symbol
// OSL_UNITTEST_VERBOSE is defined to be nonzero before this header is
// included, then it will print a "PASS" message.
//
// For a failure, a readable message is printed pinpointing the shader file
// and line and the expression that failed. Unless the shader defines
// OSL_UNITTEST_EXIT_ON_FAILURE to 0 before this header is included, a failure
// will exit the test shader after printing the error.

// Check that expression x is true/nonzero.
#define OSL_CHECK(x)                                                    \
    do {                                                                \
        if ((x)) {                                                      \
            if (OSL_UNITTEST_VERBOSE)                                   \
                printf("PASS: %s\n", #x);                               \
        } else {                                                        \
            printf("\nFAIL: %s:%d: %s\n\n", __FILE__, __LINE__, #x);    \
            if (OSL_UNITTEST_EXIT_ON_FAILURE)                           \
                exit();                                                 \
        }                                                               \
    } while (0)


// Check that two expressions are equal. For non-built-in types, this
// requires that a failmsg() function is defined for that type.
#define OSL_CHECK_EQUAL(x,y)                                                \
    do {                                                                    \
        if ((x) == (y)) {                                                   \
            if (OSL_UNITTEST_VERBOSE)                                       \
                printf("PASS: %s == %s\n", #x, #y);                         \
        } else {                                                            \
            failmsg(__FILE__, __LINE__, #x, x, #y, y);                      \
            if (OSL_UNITTEST_EXIT_ON_FAILURE)                               \
                exit();                                                     \
        }                                                                   \
    } while (0)
