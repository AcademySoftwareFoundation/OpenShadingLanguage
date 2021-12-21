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

color string2color(string val)
{
    if (val == "foo")
        return color(1,0,0);
    if (val == "bar")
        return color(0,1,0);
    return color(0,0,1);
}

float int2float(int val)
{
    return float(val)/256.0;
}

color matrix2color(matrix val)
{
    return color(val[0][0] + val[0][1] + val[0][2] + val[0][3] + val[3][0],    
                 val[1][0] + val[1][1] + val[1][2] + val[1][3] + val[3][1],
                 val[2][0] + val[2][1] + val[2][2] + val[2][3] + val[3][2]);
}

shader __OSL_CONCAT(b_, __OSL_XMACRO_SUFFIX) (
    int numStripes = 0,
    float f_in = 41,
    color c_in = 42,
    float dummy = 0,  // just to force a connection when opt is on
    output color out_string = 0,
    output float out_int = 0,
    output float out_float = 0,
    output color out_color = 0,
    output color out_matrix = 0,
    output color out_strings0 = 0,
    output color out_strings1 = 0,
    output color out_strings2 = 0,
    output float out_ints0 = 0,
    output float out_ints1 = 0,
    output float out_ints2 = 0,
    output float out_floats0 = 0,
    output float out_floats1 = 0,
    output float out_floats2 = 0,
    output color out_colors0 = 0,
    output color out_colors1 = 0,
    output color out_colors2 = 0,
    output color out_matrices0 = 0,
    output color out_matrices1 = 0,
    output color out_matrices2 = 0,
    output float out_result = 0,
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

    if (dummy < 0) {
        printf("uneachable, but needed to get connecting layers to particpate and call setmessage");
    }
    int result = 0;
    
    if ((numStripes != 0) && (int(P[1]*P[1]*(1.0 - P[0])*(2*numStripes))%2==0))
    {
        string foo_string = "magic";
        result += getmessage ("foo_string", foo_string);
        out_string = string2color(foo_string);

        int foo_int = 0;
        result += getmessage ("foo_int", foo_int);
        out_int = int2float(foo_int);

        float foo_float = 0;
        result += getmessage ("foo_float", foo_float);
        out_float = foo_float;

        color foo_color = 0;
        result += getmessage ("foo_color", foo_color);
        out_color = foo_color;

        matrix foo_matrix = 0;
        result += getmessage ("foo_matrix", foo_matrix);
        out_matrix = matrix2color(foo_matrix);
        
    
        string foo_strings[3] = { "unknown", "unknown", "unknown"};
        result += getmessage ("foo_strings", foo_strings);
        out_strings0 = string2color(foo_strings[0]);
        out_strings1 = string2color(foo_strings[1]);
        out_strings2 = string2color(foo_strings[2]);


        int foo_ints[3] = {0, 0, 0};
        result += getmessage ("foo_ints", foo_ints);
        out_ints0 = int2float(foo_ints[0]);
        out_ints1 = int2float(foo_ints[1]);
        out_ints2 = int2float(foo_ints[2]);

        float foo_floats[3] = {0.0, 0.0, 0.0};
        result += getmessage ("foo_floats", foo_floats);
        out_floats0 = foo_floats[0];
        out_floats1 = foo_floats[1];
        out_floats2 = foo_floats[2];

        color foo_colors[3] = { color(0,0,0), color(0,0,0), color(0,0,0)};
        result += getmessage ("foo_colors", foo_colors);
        out_colors0 = foo_colors[0];
        out_colors1 = foo_colors[1];
        out_colors2 = foo_colors[2];

        matrix foo_matrices[3] = {matrix(1), matrix(1), matrix(1)};
        result += getmessage ("foo_matrices", foo_matrices);
        out_matrices0 = matrix2color(foo_matrices[0]);
        out_matrices1 = matrix2color(foo_matrices[1]);
        out_matrices2 = matrix2color(foo_matrices[2]);

        int not_there = 0;
        result += getmessage ("not_there", not_there);
    }

    out_result = float(result)/20;
    
}
