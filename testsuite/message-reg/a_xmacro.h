// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_XMACRO_SUFFIX
#    error must define __OSL_XMACRO_SUFFIX to create a unique testname before including this header
#endif

#if !defined(VARYING_DATA) && !defined(UNIFORM_DATA) && !defined(CONSTANT_DATA)
#    error Must define either VARYING_DATA, UNIFORM_DATA, CONSTANT_DATA before including this xmacro!
#endif

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)

shader __OSL_CONCAT(a_, __OSL_XMACRO_SUFFIX) (
    int numStripes = 0,
    output float f_out = 0,
    output color c_out = 0,
    output float dummy = u + v  // just to force a real connection when opt is on
)
{
#ifdef CONSTANT_DATA
    float X = 0.25;
    float Y = 0.65;
#endif
#ifdef UNIFORM_DATA
    float X = 0.25;
    float Y = 0.65;
    // Intended to be unreachable, but
    // prevent constant folding of result
    if (raytype("camera") == 0) {
        X = 0.75;
        Y = 0.15;
    }
#endif
#ifdef VARYING_DATA
    float X = u;
    float Y = v;
#endif

    f_out = X;
    c_out = color (Y, X, 1);
    
    if ((numStripes != 0) && (int(P[0]*P[0]*P[1]*(2*numStripes))%2==0))
    {
        string set_string = "foo";
        if (X > 0.5)
            set_string = "bar";
        setmessage ("foo_string", set_string);

        int set_int = int(256 - (X+Y)*128);
        setmessage ("foo_int", set_int);

        float set_float = (X+Y)*0.5;
        setmessage ("foo_float", set_float);
    
        color set_color = color(1.0 - Y,X,1);
        setmessage ("foo_color", set_color);
    
        matrix set_matrix = matrix(X/16.0, Y/16.0, X/8.0, Y/8.0,
                              X/4.0, Y/4.0, X/12.0, Y/12.0,
                              X/6.0, Y/6.0, X/3.0, Y/3.0,
                              X/9.0, Y/9.0, X/10.0, Y/10.0);
        setmessage ("foo_matrix", set_matrix);


        string set_strings[3] = {set_string, "foo", "unknown"};
        setmessage ("foo_strings", set_strings);

        int set_ints[3] = {int(256 - (X+Y)*128), int(X*256), int(Y*256)};
        setmessage ("foo_ints", set_ints);
    
        float set_floats[3] = {(X+Y)*0.5, X, Y};
        setmessage ("foo_floats", set_floats);
    
        color set_colors[3] = {color(1.0 - Y,X,1), color(X,Y,(Y+X)*.05), color(Y,X,2 - (Y+X))};
        setmessage ("foo_colors", set_colors);
    
        matrix set_matrices[3] = {
            matrix(
                X/16.0, Y/16.0, X/8.0, Y/8.0,
                X/4.0, Y/4.0, X/12.0, Y/12.0,
                X/6.0, Y/6.0, X/3.0, Y/3.0,
                X/9.0, Y/9.0, X/10.0, Y/10.0
            ),
            matrix(
                X/6.0, Y/6.0, X/3.0, Y/3.0,
                X/16.0, Y/16.0, X/8.0, Y/8.0,
                X/4.0, Y/4.0, X/12.0, Y/12.0,
                X/9.0, Y/9.0, X/10.0, Y/10.0
            ),
            matrix(
                X/9.0, Y/9.0, X/10.0, Y/10.0,
                X/4.0, Y/4.0, X/12.0, Y/12.0,
                X/16.0, Y/16.0, X/8.0, Y/8.0,
                X/6.0, Y/6.0, X/3.0, Y/3.0
            )
        };
        setmessage ("foo_matrices", set_matrices);
    }
}
