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


string tostring(int x) { return format("%d", x); }
string tostring(float x) { return format("%g", x); }
string tostring(color x) { return format("%g", x); }
string tostring(point x) { return format("%g", x); }
string tostring(vector x) { return format("%g", x); }
string tostring(normal x) { return format("%g", x); }
string tostring(string x) { return x; }

#ifdef COLOR2_H
string tostring(color2 x) { return format("%g %g", x.r, x.a); }
#endif

#ifdef COLOR4_H
string tostring(color4 x)
{
    return format("%g %g %g %g", x.rgb.r, x.rgb.g, x.rgb.b, x.a);
}
#endif

#ifdef VECTOR2_H
string tostring(vector2 x) { return format("%g %g", x.x, x.y); }
#endif

#ifdef VECTOR4_H
string tostring(vector4 x)
{
    return format("%g %g %g %g", x.x, x.y, x.z, x.w);
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
// requires that a tostring() function is defined for that type.
#define OSL_CHECK_EQUAL(x,y)                                                \
    do {                                                                    \
        if ((x) == (y)) {                                                   \
            if (OSL_UNITTEST_VERBOSE)                                       \
                printf("PASS: %s == %s\n", #x, #y);                         \
        } else {                                                            \
            printf("\nFAIL: %s:%d: %s (%s) == %s (%s)\n\n",                 \
                   __FILE__, __LINE__, #x, tostring(x), #y, tostring(y));   \
            if (OSL_UNITTEST_EXIT_ON_FAILURE)                               \
                exit();                                                     \
        }                                                                   \
    } while (0)
