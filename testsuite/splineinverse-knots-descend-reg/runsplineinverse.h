// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


void init_knots (output float fltknots[16], float scale)
{
   // splineinverse requires only increasing or decreasing values
   fltknots[ 0] =  1.0 * scale;
   fltknots[ 1] =  0.95 * scale;
   fltknots[ 2] =  0.85 * scale;
   fltknots[ 3] =  0.8 * scale;
   fltknots[ 4] =  0.65 * scale;
   fltknots[ 5] =  0.59 * scale;
   fltknots[ 6] =  0.58 * scale;
   fltknots[ 7] =  0.55 * scale;
   fltknots[ 8] =  0.5 * scale;
   fltknots[ 9] =  0.3 * scale;
   fltknots[10] =  0.275 * scale;
   fltknots[11] =  0.25 * scale;
   fltknots[12] =  0.2 * scale;
   fltknots[13] =  0.1 * scale;
   fltknots[14] =  0.05 * scale;
   fltknots[15] =  0.0 * scale;
}


float run_all_fsplineinverses (
    float x,
    float scale)
{
    float fltknots[16];
    init_knots (fltknots, scale);

    // x values out of range of fltknots will not initialize r
    // so make sure to give it a valid value here
    float r = 1.0;
    if (v < (1.0/6.0)) {
        r = splineinverse("catmull-rom", x, fltknots);
    } else if (v < (2.0/6.0)) {
        r = splineinverse("bezier", x, fltknots);
    } else if (v < (3.0/6.0)) {
        r = splineinverse("bspline", x, fltknots);
    } else if (v < (4.0/6.0)) {
        r = splineinverse("hermite", x, fltknots);
    } else if (v < (5.0/6.0)) {
        r = splineinverse("linear", x, fltknots);
    } else {
        r = splineinverse("constant", x, fltknots);
    }
    return r;
}
