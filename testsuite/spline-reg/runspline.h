// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


void init_knots (output float fltknots[16], float scale)
{
   fltknots[ 0] =  0.0 * scale;
   fltknots[ 1] =  0.1 * scale;
   fltknots[ 2] =  0.4 * scale;
   fltknots[ 3] =  1.0 * scale;
   fltknots[ 4] =  0.9 * scale;
   fltknots[ 5] =  0.8 * scale;
   fltknots[ 6] =  0.7 * scale;
   fltknots[ 7] =  0.6 * scale;
   fltknots[ 8] =  0.5 * scale;
   fltknots[ 9] =  0.1 * scale;
   fltknots[10] =  0.0 * scale;
   fltknots[11] =  0.3 * scale;
   fltknots[12] =  0.8 * scale;
   fltknots[13] =  0.4 * scale;
   fltknots[14] =  0.2 * scale;
   fltknots[15] =  0.0 * scale;
}



float run_all_fsplines (
    float x,
    float scale)
{
    float fltknots[16];
    init_knots (fltknots, scale);

    float r;
    if (v < (1.0/6.0)) {
        r = spline("catmull-rom", x, fltknots);
    } else if (v < (2.0/6.0)) {
        r = spline("bezier", x, fltknots);
    } else if (v < (3.0/6.0)) {
        r = spline("bspline", x, fltknots);
    } else if (v < (4.0/6.0)) {
        r = spline("hermite", x, fltknots);
    } else if (v < (5.0/6.0)) {
        r = spline("linear", x, fltknots);
    } else {
        r = spline("constant", x, fltknots);
    }
    return r;
}

void init_knots (output color clrknots[16], color scale)
{
    clrknots[ 0] =  color(0.0) * scale;
    clrknots[ 1] =  color(0.1) * scale;
    clrknots[ 2] =  color(0.4) * scale;
    clrknots[ 3] =  color(1.0) * scale;
    clrknots[ 4] =  color(0.9) * scale;
    clrknots[ 5] =  color(0.8) * scale;
    clrknots[ 6] =  color(0.7) * scale;
    clrknots[ 7] =  color(0.6) * scale;
    clrknots[ 8] =  color(0.5) * scale;
    clrknots[ 9] =  color(0.1) * scale;
    clrknots[10] =  color(0.0) * scale;
    clrknots[11] =  color(0.3) * scale;
    clrknots[12] =  color(0.8) * scale;
    clrknots[13] =  color(0.4) * scale;
    clrknots[14] =  color(0.2) * scale;
    clrknots[15] =  color(0.0) * scale;
}

color run_all_csplines (
    float x,
    color scale)
{
    color clrknots[16];
    init_knots (clrknots, scale);

    color r;
    if (v < (1.0/6.0)) {
        r = spline("catmull-rom", x, clrknots);
    } else if (v < (2.0/6.0)) {
        r = spline("bezier", x, clrknots);
    } else if (v < (3.0/6.0)) {
        r = spline("bspline", x, clrknots);
    } else if (v < (4.0/6.0)) {
        r = spline("hermite", x, clrknots);
    } else if (v < (5.0/6.0)) {
        r = spline("linear", x, clrknots);
    } else {
        r = spline("constant", x, clrknots);
    }
    return r;
}

color run_all_csplines_knots_with_no_derivs (
    float x,
    vector scale)
{
    // Only way discovered to get varying non-derivative
    // knot values is to make the call ambiguous
    // and just pass a triple instead of a proper
    // array
    color r;
    if (v < (1.0/6.0)) {
        r = spline("catmull-rom", x, scale);
    } else if (v < (2.0/6.0)) {
        r = spline("bezier", x, scale);
    } else if (v < (3.0/6.0)) {
        r = spline("bspline", x, scale);
    } else if (v < (4.0/6.0)) {
        r = spline("hermite", x, scale);
    } else if (v < (5.0/6.0)) {
        r = spline("linear", x, scale);
    } else {
        r = spline("constant", x, scale);
    }
    return r;
}

