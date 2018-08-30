/*
Copyright (c) 2013 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



// The simplex noise code was adapted from code by Stefan Gustavson,
//   http://staffwww.itn.liu.se/~stegu/aqsis/aqsis-newnoise/sdnoise1234.c
//
// For incorporation into OSL, it's been heavily modified and optimized
// by Larry Gritz, including in the following ways:
//
// * Instead of permutation tables (which make it inherently periodic),
//   use a hash to scramble the simplex corner lookups.  Not only does
//   this make the pattern effectively aperiodic, but it's also faster!
//   (presumably because the hash can be done faster than the table
//   lookup)
// * Added 'seed' parameters to more easily generate 'vector' versions
//   of the noise, where each component starts with a differen seed.
// * Change the gradN functions to return float* into the array, rather
//   have a float* parameter for each dimension.
// * Simplify the function logic to have fewer assignments, faster.
//


// This was Stefan Gustavson's original copyright notice:
//
/* sdnoise1234, Simplex noise with true analytic
 * derivative in 1D to 4D.
 *
 * Copyright © 2003-2011, Stefan Gustavson
 *
 * Contact: stefan.gustavson@gmail.com
 *
 * This library is public domain software, released by the author
 * into the public domain in February 2011. You may do anything
 * you like with it. You may even remove all attributions,
 * but of course I'd appreciate it if you kept my name somewhere.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 */



#include <cmath>
#include <iostream>

#include <OSL/oslnoise.h>
#include <OpenImageIO/fmath.h>

OSL_NAMESPACE_ENTER

namespace pvt {


inline OSL_HOSTDEVICE uint32_t
scramble (uint32_t v0, uint32_t v1=0, uint32_t v2=0)
{
    return OIIO::bjhash::bjfinal (v0, v1, v2^0xdeadbeef);
}



/* Static data ---------------------- */

static OSL_DEVICE float zero[] = { 0.0f, 0.0f, 0.0f, 0.0f };

// Gradient table for 2D. These could be programmed the Ken Perlin way with
// some clever bit-twiddling, but this is more clear, and not really slower.
static OSL_DEVICE float grad2lut[8][2] = {
    { -1.0f, -1.0f }, { 1.0f,  0.0f }, { -1.0f, 0.0f }, { 1.0f,  1.0f },
    { -1.0f,  1.0f }, { 0.0f, -1.0f }, {  0.0f, 1.0f }, { 1.0f, -1.0f }
};

// Gradient directions for 3D.
// These vectors are based on the midpoints of the 12 edges of a cube.
// A larger array of random unit length vectors would also do the job,
// but these 12 (including 4 repeats to make the array length a power
// of two) work better. They are not random, they are carefully chosen
// to represent a small, isotropic set of directions.
static OSL_DEVICE float grad3lut[16][3] = {
    {  1.0f,  0.0f,  1.0f }, {  0.0f,  1.0f,  1.0f }, // 12 cube edges
    { -1.0f,  0.0f,  1.0f }, {  0.0f, -1.0f,  1.0f },
    {  1.0f,  0.0f, -1.0f }, {  0.0f,  1.0f, -1.0f },
    { -1.0f,  0.0f, -1.0f }, {  0.0f, -1.0f, -1.0f },
    {  1.0f, -1.0f,  0.0f }, {  1.0f,  1.0f,  0.0f },
    { -1.0f,  1.0f,  0.0f }, { -1.0f, -1.0f,  0.0f },
    {  1.0f,  0.0f,  1.0f }, { -1.0f,  0.0f,  1.0f }, // 4 repeats to make 16
    {  0.0f,  1.0f, -1.0f }, {  0.0f, -1.0f, -1.0f }
};

// Gradient directions for 4D
static OSL_DEVICE float grad4lut[32][4] = {
  { 0.0f, 1.0f, 1.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, -1.0f }, { 0.0f, 1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f, -1.0f, -1.0f }, // 32 tesseract edges
  { 0.0f, -1.0f, 1.0f, 1.0f }, { 0.0f, -1.0f, 1.0f, -1.0f }, { 0.0f, -1.0f, -1.0f, 1.0f }, { 0.0f, -1.0f, -1.0f, -1.0f },
  { 1.0f, 0.0f, 1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f, -1.0f }, { 1.0f, 0.0f, -1.0f, 1.0f }, { 1.0f, 0.0f, -1.0f, -1.0f },
  { -1.0f, 0.0f, 1.0f, 1.0f }, { -1.0f, 0.0f, 1.0f, -1.0f }, { -1.0f, 0.0f, -1.0f, 1.0f }, { -1.0f, 0.0f, -1.0f, -1.0f },
  { 1.0f, 1.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 0.0f, -1.0f }, { 1.0f, -1.0f, 0.0f, 1.0f }, { 1.0f, -1.0f, 0.0f, -1.0f },
  { -1.0f, 1.0f, 0.0f, 1.0f }, { -1.0f, 1.0f, 0.0f, -1.0f }, { -1.0f, -1.0f, 0.0f, 1.0f }, { -1.0f, -1.0f, 0.0f, -1.0f },
  { 1.0f, 1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, -1.0f, 0.0f }, { 1.0f, -1.0f, 1.0f, 0.0f }, { 1.0f, -1.0f, -1.0f, 0.0f },
  { -1.0f, 1.0f, 1.0f, 0.0f }, { -1.0f, 1.0f, -1.0f, 0.0f }, { -1.0f, -1.0f, 1.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, 0.0f }
};

// A lookup table to traverse the simplex around a given point in 4D.
// Details can be found where this table is used, in the 4D noise method.
/* TODO: This should not be required, backport it from Bill's GLSL code! */
static OSL_DEVICE unsigned char simplex[64][4] = {
  {0,1,2,3},{0,1,3,2},{0,0,0,0},{0,2,3,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,2,3,0},
  {0,2,1,3},{0,0,0,0},{0,3,1,2},{0,3,2,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,3,2,0},
  {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
  {1,2,0,3},{0,0,0,0},{1,3,0,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,3,0,1},{2,3,1,0},
  {1,0,2,3},{1,0,3,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,0,3,1},{0,0,0,0},{2,1,3,0},
  {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
  {2,0,1,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,0,1,2},{3,0,2,1},{0,0,0,0},{3,1,2,0},
  {2,1,0,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,1,0,2},{0,0,0,0},{3,2,0,1},{3,2,1,0}};

/* --------------------------------------------------------------------- */

/*
 * Helper functions to compute gradients in 1D to 4D
 * and gradients-dot-residualvectors in 2D to 4D.
 */

inline OSL_HOSTDEVICE float grad1 (int i, int seed)
{
    int h = scramble (i, seed);
    float g = 1.0f + (h & 7);   // Gradient value is one of 1.0, 2.0, ..., 8.0
    if (h & 8)
        g = -g;   // Make half of the gradients negative
    return g;
}

inline OSL_HOSTDEVICE const float * grad2 (int i, int j, int seed)
{
    int h = scramble (i, j, seed);
    return grad2lut[h & 7];
}

inline OSL_HOSTDEVICE const float * grad3 (int i, int j, int k, int seed)
{
    int h = scramble (i, j, scramble (k, seed));
    return grad3lut[h & 15];
}

inline OSL_HOSTDEVICE const float * grad4 (int i, int j, int k, int l, int seed)
{
    int h = scramble (i, j, scramble (k, l, seed));
    return grad4lut[h & 31];
}


// 1D simplex noise with derivative.
// If the last argument is not null, the analytic derivative
// is also calculated.
OSL_HOSTDEVICE float
simplexnoise1 (float x, int seed, float *dnoise_dx)
{
    int i0 = OIIO::ifloor(x);
    int i1 = i0 + 1;
    float x0 = x - i0;
    float x1 = x0 - 1.0f;

    float x20 = x0*x0;
    float t0 = 1.0f - x20;
    //  if(t0 < 0.0f) t0 = 0.0f; // Never happens for 1D: x0<=1 always
    float t20 = t0 * t0;
    float t40 = t20 * t20;
    float gx0 = grad1 (i0, seed);
    float n0 = t40 * gx0 * x0;

    float x21 = x1*x1;
    float t1 = 1.0f - x21;
    //  if(t1 < 0.0f) t1 = 0.0f; // Never happens for 1D: |x1|<=1 always
    float t21 = t1 * t1;
    float t41 = t21 * t21;
    float gx1 = grad1 (i1, seed);
    float n1 = t41 * gx1 * x1;

    // Sum up and scale the result.  The scale is empirical, to make it
    // cover [-1,1], and to make it approximately match the range of our
    // Perlin noise implementation.
    const float scale = 0.36f;

    if (dnoise_dx) {
        // Compute derivative according to:
        // *dnoise_dx = -8.0f * t20 * t0 * x0 * (gx0 * x0) + t40 * gx0;
        // *dnoise_dx += -8.0f * t21 * t1 * x1 * (gx1 * x1) + t41 * gx1;
        *dnoise_dx = t20 * t0 * gx0 * x20;
        *dnoise_dx += t21 * t1 * gx1 * x21;
        *dnoise_dx *= -8.0f;
        *dnoise_dx += t40 * gx0 + t41 * gx1;
        *dnoise_dx *= scale;
    }

    return scale * (n0 + n1);
}



// 2D simplex noise with derivatives.
// If the last two arguments are not null, the analytic derivative
// (the 2D gradient of the scalar noise field) is also calculated.
OSL_HOSTDEVICE float simplexnoise2 (float x, float y, int seed,
                                    float *dnoise_dx, float *dnoise_dy)
{
    // Skewing factors for 2D simplex grid:
    const float F2 = 0.366025403;   // = 0.5*(sqrt(3.0)-1.0)
    const float G2 = 0.211324865;  // = (3.0-Math.sqrt(3.0))/6.0
    const float *g0 = zero, *g1 = zero, *g2 = zero;

    /* Skew the input space to determine which simplex cell we're in */
    float s = ( x + y ) * F2; /* Hairy factor for 2D */
    float xs = x + s;
    float ys = y + s;
    int i = OIIO::ifloor(xs);
    int j = OIIO::ifloor(ys);

    float t = (float) (i + j) * G2;
    float X0 = i - t; /* Unskew the cell origin back to (x,y) space */
    float Y0 = j - t;
    float x0 = x - X0; /* The x,y distances from the cell origin */
    float y0 = y - Y0;

    /* For the 2D case, the simplex shape is an equilateral triangle.
     * Determine which simplex we are in. */
    int i1, j1; // Offsets for second (middle) corner of simplex in (i,j) coords
    if (x0 > y0) {
        i1 = 1; j1 = 0;   // lower triangle, XY order: (0,0)->(1,0)->(1,1)
    } else {
        i1 = 0; j1 = 1;   // upper triangle, YX order: (0,0)->(0,1)->(1,1)
    }

    // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    // c = (3-sqrt(3))/6  
    float x1 = x0 - i1 + G2; // Offsets for middle corner in (x,y) unskewed coords
    float y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2; // Offsets for last corner in (x,y) unskewed coords
    float y2 = y0 - 1.0f + 2.0f * G2;


    // Calculate the contribution from the three corners
    float t20 = 0.0f, t40 = 0.0f;
    float t21 = 0.0f, t41 = 0.0f;
    float t22 = 0.0f, t42 = 0.0f;
    float n0=0.0f, n1=0.0f, n2=0.0f; // Noise contributions from the simplex corners

    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 >= 0.0f) {
        g0 = grad2 (i, j, seed);
        t20 = t0 * t0;
        t40 = t20 * t20;
        n0 = t40 * (g0[0] * x0 + g0[1] * y0);
    }

    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 >= 0.0f) {
        g1 = grad2 (i+i1, j+j1, seed);
        t21 = t1 * t1;
        t41 = t21 * t21;
        n1 = t41 * (g1[0] * x1 + g1[1] * y1);
    }

    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 >= 0.0f) {
        g2 = grad2 (i+1, j+1, seed);
        t22 = t2 * t2;
        t42 = t22 * t22;
        n2 = t42 * (g2[0] * x2 + g2[1] * y2);
    }

    // Sum up and scale the result.  The scale is empirical, to make it
    // cover [-1,1], and to make it approximately match the range of our
    // Perlin noise implementation.
    const float scale = 64.0f;
    float noise = scale * (n0 + n1 + n2);

    // Compute derivative, if requested by supplying non-null pointers
    // for the last two arguments
    if (dnoise_dx) {
        DASSERT (dnoise_dy);
	/*  A straight, unoptimised calculation would be like:
     *    *dnoise_dx = -8.0f * t20 * t0 * x0 * ( g0[0] * x0 + g0[1] * y0 ) + t40 * g0[0];
     *    *dnoise_dy = -8.0f * t20 * t0 * y0 * ( g0[0] * x0 + g0[1] * y0 ) + t40 * g0[1];
     *    *dnoise_dx += -8.0f * t21 * t1 * x1 * ( g1[0] * x1 + g1[1] * y1 ) + t41 * g1[0];
     *    *dnoise_dy += -8.0f * t21 * t1 * y1 * ( g1[0] * x1 + g1[1] * y1 ) + t41 * g1[1];
     *    *dnoise_dx += -8.0f * t22 * t2 * x2 * ( g2[0] * x2 + g2[1] * y2 ) + t42 * g2[0];
     *    *dnoise_dy += -8.0f * t22 * t2 * y2 * ( g2[0] * x2 + g2[1] * y2 ) + t42 * g2[1];
	 */
        float temp0 = t20 * t0 * (g0[0]* x0 + g0[1] * y0);
        *dnoise_dx = temp0 * x0;
        *dnoise_dy = temp0 * y0;
        float temp1 = t21 * t1 * (g1[0] * x1 + g1[1] * y1);
        *dnoise_dx += temp1 * x1;
        *dnoise_dy += temp1 * y1;
        float temp2 = t22 * t2 * (g2[0]* x2 + g2[1] * y2);
        *dnoise_dx += temp2 * x2;
        *dnoise_dy += temp2 * y2;
        *dnoise_dx *= -8.0f;
        *dnoise_dy *= -8.0f;
        *dnoise_dx += t40 * g0[0] + t41 * g1[0] + t42 * g2[0];
        *dnoise_dy += t40 * g0[1] + t41 * g1[1] + t42 * g2[1];
        *dnoise_dx *= scale; /* Scale derivative to match the noise scaling */
        *dnoise_dy *= scale;
    }
    return noise;
}



// 3D simplex noise with derivatives.
// If the last tthree arguments are not null, the analytic derivative
// (the 3D gradient of the scalar noise field) is also calculated.
OSL_HOSTDEVICE float
simplexnoise3 (float x, float y, float z, int seed,
               float *dnoise_dx, float *dnoise_dy, float *dnoise_dz)
{
    // Skewing factors for 3D simplex grid:
    const float F3 = 0.333333333;   // = 1/3
    const float G3 = 0.166666667;   // = 1/6

    // Gradients at simplex corners
    const float *g0 = zero, *g1 = zero, *g2 = zero, *g3 = zero;

    // Skew the input space to determine which simplex cell we're in
    float s = (x+y+z)*F3; // Very nice and simple skew factor for 3D
    float xs = x+s;
    float ys = y+s;
    float zs = z+s;
    int i = OIIO::ifloor(xs);
    int j = OIIO::ifloor(ys);
    int k = OIIO::ifloor(zs);

    float t = (float)(i+j+k)*G3; 
    float X0 = i-t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = j-t;
    float Z0 = k-t;
    float x0 = x-X0; // The x,y,z distances from the cell origin
    float y0 = y-Y0;
    float z0 = z-Z0;

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords

#if 1
    // TODO: This code would benefit from a backport from the GLSL version!
    // (no it can't... see note below)
    if (x0>=y0) {
        if (y0>=z0) {
            i1=1; j1=0; k1=0; i2=1; j2=1; k2=0;  /* X Y Z order */
        } else if (x0>=z0) {
            i1=1; j1=0; k1=0; i2=1; j2=0; k2=1;  /* X Z Y order */
        } else {
            i1=0; j1=0; k1=1; i2=1; j2=0; k2=1;  /* Z X Y order */
        }
    } else { // x0<y0
        if (y0<z0) {
            i1=0; j1=0; k1=1; i2=0; j2=1; k2=1;  /* Z Y X order */
        } else if (x0<z0) {
            i1=0; j1=1; k1=0; i2=0; j2=1; k2=1;  /* Y Z X order */
        } else {
            i1=0; j1=1; k1=0; i2=1; j2=1; k2=0;  /* Y X Z order */
        }
    }
#else
    // Here's the logic "from the GLSL version", near as I (LG) could
    // translate it from GLSL to non-SIMD C++.  It was slower.  I'm
    // keeping this code here for reference anyway.
    {
        // vec3 g = step(x0.yzx, x0.xyz);
        // vec3 l = 1.0 - g;
        bool g0 = (x0 >= y0), l0 = !g0;
        bool g1 = (y0 >= z0), l1 = !g1;
        bool g2 = (z0 >= x0), l2 = !g2;
        // vec3 i1 = min (g.xyz, l.zxy);  // min of bools is &
        // vec3 i2 = max (g.xyz, l.zxy);  // max of bools is |
        i1 = g0 & l2;
        j1 = g1 & l0;
        k1 = g2 & l1;
        i2 = g0 | l2;
        j2 = g1 | l0;
        k2 = g2 | l1;
    }
#endif

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z),
    // where c = 1/6.
    float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    float t20 = 0.0f, t40 = 0.0f;
    float t21 = 0.0f, t41 = 0.0f;
    float t22 = 0.0f, t42 = 0.0f;
    float t23 = 0.0f, t43 = 0.0f;
    float n0=0.0f, n1=0.0f, n2=0.0f, n3=0.0f; // Noise contributions from the four simplex corners

    // Calculate the contribution from the four corners
    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
    if (t0 >= 0.0f) {
        g0 = grad3 (i, j, k, seed);
        t20 = t0 * t0;
        t40 = t20 * t20;
        n0 = t40 * (g0[0] * x0 + g0[1] * y0 + g0[2] * z0);
    }

    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
    if (t1 >= 0.0f) {
        g1 = grad3 (i+i1, j+j1, k+k1, seed);
        t21 = t1 * t1;
        t41 = t21 * t21;
        n1 = t41 * (g1[0] * x1 + g1[1] * y1 + g1[2] * z1);
    }

    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
    if (t2 >= 0.0f) {
        g2 = grad3 (i+i2, j+j2, k+k2, seed);
        t22 = t2 * t2;
        t42 = t22 * t22;
        n2 = t42 * (g2[0] * x2 + g2[1] * y2 + g2[2] * z2);
    }

    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
    if (t3 >= 0.0f) {
        g3 = grad3 (i+1, j+1, k+1, seed);
        t23 = t3 * t3;
        t43 = t23 * t23;
        n3 = t43 * (g3[0] * x3 + g3[1] * y3 + g3[2] * z3);
    }

    // Sum up and scale the result.  The scale is empirical, to make it
    // cover [-1,1], and to make it approximately match the range of our
    // Perlin noise implementation.
    const float scale = 68.0f;
    float noise = scale * (n0 + n1 + n2 + n3);

    // Compute derivative, if requested by supplying non-null pointers
    // for the last three arguments
    if (dnoise_dx) {
        DASSERT (dnoise_dy && dnoise_dz);
	/*  A straight, unoptimised calculation would be like:
     *     *dnoise_dx = -8.0f * t20 * t0 * x0 * dot(g0[0], g0[1], g0[2], x0, y0, z0) + t40 * g0[0];
     *    *dnoise_dy = -8.0f * t20 * t0 * y0 * dot(g0[0], g0[1], g0[2], x0, y0, z0) + t40 * g0[1];
     *    *dnoise_dz = -8.0f * t20 * t0 * z0 * dot(g0[0], g0[1], g0[2], x0, y0, z0) + t40 * g0[2];
     *    *dnoise_dx += -8.0f * t21 * t1 * x1 * dot(g1[0], g1[1], g1[2], x1, y1, z1) + t41 * g1[0];
     *    *dnoise_dy += -8.0f * t21 * t1 * y1 * dot(g1[0], g1[1], g1[2], x1, y1, z1) + t41 * g1[1];
     *    *dnoise_dz += -8.0f * t21 * t1 * z1 * dot(g1[0], g1[1], g1[2], x1, y1, z1) + t41 * g1[2];
     *    *dnoise_dx += -8.0f * t22 * t2 * x2 * dot(g2[0], g2[1], g2[2], x2, y2, z2) + t42 * g2[0];
     *    *dnoise_dy += -8.0f * t22 * t2 * y2 * dot(g2[0], g2[1], g2[2], x2, y2, z2) + t42 * g2[1];
     *    *dnoise_dz += -8.0f * t22 * t2 * z2 * dot(g2[0], g2[1], g2[2], x2, y2, z2) + t42 * g2[2];
     *    *dnoise_dx += -8.0f * t23 * t3 * x3 * dot(g3[0], g3[1], g3[2], x3, y3, z3) + t43 * g3[0];
     *    *dnoise_dy += -8.0f * t23 * t3 * y3 * dot(g3[0], g3[1], g3[2], x3, y3, z3) + t43 * g3[1];
     *    *dnoise_dz += -8.0f * t23 * t3 * z3 * dot(g3[0], g3[1], g3[2], x3, y3, z3) + t43 * g3[2];
     */
        float temp0 = t20 * t0 * (g0[0] * x0 + g0[1] * y0 + g0[2] * z0);
        *dnoise_dx = temp0 * x0;
        *dnoise_dy = temp0 * y0;
        *dnoise_dz = temp0 * z0;
        float temp1 = t21 * t1 * (g1[0] * x1 + g1[1] * y1 + g1[2] * z1);
        *dnoise_dx += temp1 * x1;
        *dnoise_dy += temp1 * y1;
        *dnoise_dz += temp1 * z1;
        float temp2 = t22 * t2 * (g2[0] * x2 + g2[1] * y2 + g2[2] * z2);
        *dnoise_dx += temp2 * x2;
        *dnoise_dy += temp2 * y2;
        *dnoise_dz += temp2 * z2;
        float temp3 = t23 * t3 * (g3[0] * x3 + g3[1] * y3 + g3[2] * z3);
        *dnoise_dx += temp3 * x3;
        *dnoise_dy += temp3 * y3;
        *dnoise_dz += temp3 * z3;
        *dnoise_dx *= -8.0f;
        *dnoise_dy *= -8.0f;
        *dnoise_dz *= -8.0f;
        *dnoise_dx += t40 * g0[0] + t41 * g1[0] + t42 * g2[0] + t43 * g3[0];
        *dnoise_dy += t40 * g0[1] + t41 * g1[1] + t42 * g2[1] + t43 * g3[1];
        *dnoise_dz += t40 * g0[2] + t41 * g1[2] + t42 * g2[2] + t43 * g3[2];
        *dnoise_dx *= scale; // Scale derivative to match the noise scaling
        *dnoise_dy *= scale;
        *dnoise_dz *= scale;
    }

    return noise;
}



// 4D simplex noise with derivatives.
// If the last four arguments are not null, the analytic derivative
// (the 4D gradient of the scalar noise field) is also calculated.
OSL_HOSTDEVICE float
simplexnoise4 (float x, float y, float z, float w, int seed,
               float *dnoise_dx, float *dnoise_dy,
               float *dnoise_dz, float *dnoise_dw)
{
    // The skewing and unskewing factors are hairy again for the 4D case
    const float F4 = 0.309016994; // F4 = (Math.sqrt(5.0)-1.0)/4.0
    const float G4 = 0.138196601; // G4 = (5.0-Math.sqrt(5.0))/20.0

    // Gradients at simplex corners
    const float *g0 = zero, *g1 = zero, *g2 = zero, *g3 = zero, *g4 = zero;

    // Noise contributions from the four simplex corners
    float n0=0.0f, n1=0.0f, n2=0.0f, n3=0.0f, n4=0.0f;
    float t20 = 0.0f, t21 = 0.0f, t22 = 0.0f, t23 = 0.0f, t24 = 0.0f;
    float t40 = 0.0f, t41 = 0.0f, t42 = 0.0f, t43 = 0.0f, t44 = 0.0f;

    // Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
    float s = (x + y + z + w) * F4; // Factor for 4D skewing
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    float ws = w + s;
    int i = OIIO::ifloor(xs);
    int j = OIIO::ifloor(ys);
    int k = OIIO::ifloor(zs);
    int l = OIIO::ifloor(ws);

    float t = (i + j + k + l) * G4; // Factor for 4D unskewing
    float X0 = i - t; // Unskew the cell origin back to (x,y,z,w) space
    float Y0 = j - t;
    float Z0 = k - t;
    float W0 = l - t;

    float x0 = x - X0;  // The x,y,z,w distances from the cell origin
    float y0 = y - Y0;
    float z0 = z - Z0;
    float w0 = w - W0;

    // For the 4D case, the simplex is a 4D shape I won't even try to describe.
    // To find out which of the 24 possible simplices we're in, we need to
    // determine the magnitude ordering of x0, y0, z0 and w0.
    // The method below is a reasonable way of finding the ordering of x,y,z,w
    // and then find the correct traversal order for the simplex we’re in.
    // First, six pair-wise comparisons are performed between each possible pair
    // of the four coordinates, and then the results are used to add up binary
    // bits for an integer index into a precomputed lookup table, simplex[].
    int c1 = (x0 > y0) ? 32 : 0;
    int c2 = (x0 > z0) ? 16 : 0;
    int c3 = (y0 > z0) ? 8 : 0;
    int c4 = (x0 > w0) ? 4 : 0;
    int c5 = (y0 > w0) ? 2 : 0;
    int c6 = (z0 > w0) ? 1 : 0;
    int c = c1 | c2 | c3 | c4 | c5 | c6; // '|' is mostly faster than '+'

    int i1, j1, k1, l1; // The integer offsets for the second simplex corner
    int i2, j2, k2, l2; // The integer offsets for the third simplex corner
    int i3, j3, k3, l3; // The integer offsets for the fourth simplex corner

    // simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
    // Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
    // impossible. Only the 24 indices which have non-zero entries make any sense.
    // We use a thresholding to set the coordinates in turn from the largest magnitude.
    // The number 3 in the "simplex" array is at the position of the largest coordinate.
    i1 = simplex[c][0]>=3 ? 1 : 0;
    j1 = simplex[c][1]>=3 ? 1 : 0;
    k1 = simplex[c][2]>=3 ? 1 : 0;
    l1 = simplex[c][3]>=3 ? 1 : 0;
    // The number 2 in the "simplex" array is at the second largest coordinate.
    i2 = simplex[c][0]>=2 ? 1 : 0;
    j2 = simplex[c][1]>=2 ? 1 : 0;
    k2 = simplex[c][2]>=2 ? 1 : 0;
    l2 = simplex[c][3]>=2 ? 1 : 0;
    // The number 1 in the "simplex" array is at the second smallest coordinate.
    i3 = simplex[c][0]>=1 ? 1 : 0;
    j3 = simplex[c][1]>=1 ? 1 : 0;
    k3 = simplex[c][2]>=1 ? 1 : 0;
    l3 = simplex[c][3]>=1 ? 1 : 0;
    // The fifth corner has all coordinate offsets = 1, so no need to look that up.

    float x1 = x0 - i1 + G4; // Offsets for second corner in (x,y,z,w) coords
    float y1 = y0 - j1 + G4;
    float z1 = z0 - k1 + G4;
    float w1 = w0 - l1 + G4;
    float x2 = x0 - i2 + 2.0f * G4; // Offsets for third corner in (x,y,z,w) coords
    float y2 = y0 - j2 + 2.0f * G4;
    float z2 = z0 - k2 + 2.0f * G4;
    float w2 = w0 - l2 + 2.0f * G4;
    float x3 = x0 - i3 + 3.0f * G4; // Offsets for fourth corner in (x,y,z,w) coords
    float y3 = y0 - j3 + 3.0f * G4;
    float z3 = z0 - k3 + 3.0f * G4;
    float w3 = w0 - l3 + 3.0f * G4;
    float x4 = x0 - 1.0f + 4.0f * G4; // Offsets for last corner in (x,y,z,w) coords
    float y4 = y0 - 1.0f + 4.0f * G4;
    float z4 = z0 - 1.0f + 4.0f * G4;
    float w4 = w0 - 1.0f + 4.0f * G4;

    // Calculate the contribution from the five corners
    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0 - w0*w0;
    if (t0 >= 0.0f) {
        t20 = t0 * t0;
        t40 = t20 * t20;
        g0 = grad4 (i, j, k, l, seed);
        n0 = t40 * (g0[0] * x0 + g0[1] * y0 + g0[2] * z0 + g0[3] * w0);
    }

    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1 - w1*w1;
    if (t1 >= 0.0f) {
        t21 = t1 * t1;
        t41 = t21 * t21;
        g1 = grad4 (i+i1, j+j1, k+k1, l+l1, seed);
        n1 = t41 * (g1[0] * x1 + g1[1] * y1 + g1[2] * z1 + g1[3] * w1);
    }

    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2 - w2*w2;
    if (t2 >= 0.0f) {
        t22 = t2 * t2;
        t42 = t22 * t22;
        g2 = grad4 (i+i2, j+j2, k+k2, l+l2, seed);
        n2 = t42 * (g2[0] * x2 + g2[1] * y2 + g2[2] * z2 + g2[3] * w2);
   }

    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3 - w3*w3;
    if (t3 >= 0.0f) {
        t23 = t3 * t3;
        t43 = t23 * t23;
        g3 = grad4 (i+i3, j+j3, k+k3, l+l3, seed);
        n3 = t43 * (g3[0] * x3 + g3[1] * y3 + g3[2] * z3 + g3[3] * w3);
    }

    float t4 = 0.5f - x4*x4 - y4*y4 - z4*z4 - w4*w4;
    if (t4 >= 0.0f) {
        t24 = t4 * t4;
        t44 = t24 * t24;
        g4 = grad4 (i+1, j+1, k+1, l+1, seed);
        n4 = t44 * (g4[0] * x4 + g4[1] * y4 + g4[2] * z4 + g4[3] * w4);
    }

    // Sum up and scale the result.  The scale is empirical, to make it
    // cover [-1,1], and to make it approximately match the range of our
    // Perlin noise implementation.
    const float scale = 54.0f;
    float noise = scale * (n0 + n1 + n2 + n3 + n4);

    // Compute derivative, if requested by supplying non-null pointers
    // for the last four arguments
    if (dnoise_dx) {
        DASSERT (dnoise_dy && dnoise_dz && dnoise_dw);
	/*  A straight, unoptimised calculation would be like:
     *     *dnoise_dx = -8.0f * t20 * t0 * x0 * dot(g0[0], g0[1], g0[2], g0[3], x0, y0, z0, w0) + t40 * g0[0];
     *    *dnoise_dy = -8.0f * t20 * t0 * y0 * dot(g0[0], g0[1], g0[2], g0[3], x0, y0, z0, w0) + t40 * g0[1];
     *    *dnoise_dz = -8.0f * t20 * t0 * z0 * dot(g0[0], g0[1], g0[2], g0[3], x0, y0, z0, w0) + t40 * g0[2];
     *    *dnoise_dw = -8.0f * t20 * t0 * w0 * dot(g0[0], g0[1], g0[2], g0[3], x0, y0, z0, w0) + t40 * g0[3];
     *    *dnoise_dx += -8.0f * t21 * t1 * x1 * dot(g1[0], g1[1], g1[2], g1[3], x1, y1, z1, w1) + t41 * g1[0];
     *    *dnoise_dy += -8.0f * t21 * t1 * y1 * dot(g1[0], g1[1], g1[2], g1[3], x1, y1, z1, w1) + t41 * g1[1];
     *    *dnoise_dz += -8.0f * t21 * t1 * z1 * dot(g1[0], g1[1], g1[2], g1[3], x1, y1, z1, w1) + t41 * g1[2];
     *    *dnoise_dw = -8.0f * t21 * t1 * w1 * dot(g1[0], g1[1], g1[2], g1[3], x1, y1, z1, w1) + t41 * g1[3];
     *    *dnoise_dx += -8.0f * t22 * t2 * x2 * dot(g2[0], g2[1], g2[2], g2[3], x2, y2, z2, w2) + t42 * g2[0];
     *    *dnoise_dy += -8.0f * t22 * t2 * y2 * dot(g2[0], g2[1], g2[2], g2[3], x2, y2, z2, w2) + t42 * g2[1];
     *    *dnoise_dz += -8.0f * t22 * t2 * z2 * dot(g2[0], g2[1], g2[2], g2[3], x2, y2, z2, w2) + t42 * g2[2];
     *    *dnoise_dw += -8.0f * t22 * t2 * w2 * dot(g2[0], g2[1], g2[2], g2[3], x2, y2, z2, w2) + t42 * g2[3];
     *    *dnoise_dx += -8.0f * t23 * t3 * x3 * dot(g3[0], g3[1], g3[2], g3[3], x3, y3, z3, w3) + t43 * g3[0];
     *    *dnoise_dy += -8.0f * t23 * t3 * y3 * dot(g3[0], g3[1], g3[2], g3[3], x3, y3, z3, w3) + t43 * g3[1];
     *    *dnoise_dz += -8.0f * t23 * t3 * z3 * dot(g3[0], g3[1], g3[2], g3[3], x3, y3, z3, w3) + t43 * g3[2];
     *    *dnoise_dw += -8.0f * t23 * t3 * w3 * dot(g3[0], g3[1], g3[2], g3[3], x3, y3, z3, w3) + t43 * g3[3];
     *    *dnoise_dx += -8.0f * t24 * t4 * x4 * dot(g4[0], g4[1], g4[2], g4[3], x4, y4, z4, w4) + t44 * g4[0];
     *    *dnoise_dy += -8.0f * t24 * t4 * y4 * dot(g4[0], g4[1], g4[2], g4[3], x4, y4, z4, w4) + t44 * g4[1];
     *    *dnoise_dz += -8.0f * t24 * t4 * z4 * dot(g4[0], g4[1], g4[2], g4[3], x4, y4, z4, w4) + t44 * g4[2];
     *    *dnoise_dw += -8.0f * t24 * t4 * w4 * dot(g4[0], g4[1], g4[2], g4[3], x4, y4, z4, w4) + t44 * g4[3];
     */
        float temp0 = t20 * t0 * (g0[0] * x0 + g0[1] * y0 + g0[2] * z0 + g0[3] * w0);
        *dnoise_dx = temp0 * x0;
        *dnoise_dy = temp0 * y0;
        *dnoise_dz = temp0 * z0;
        *dnoise_dw = temp0 * w0;
        float temp1 = t21 * t1 * (g1[0] * x1 + g1[1] * y1 + g1[2] * z1 + g1[3] * w1);
        *dnoise_dx += temp1 * x1;
        *dnoise_dy += temp1 * y1;
        *dnoise_dz += temp1 * z1;
        *dnoise_dw += temp1 * w1;
        float temp2 = t22 * t2 * (g2[0] * x2 + g2[1] * y2 + g2[2] * z2 + g2[3] * w2);
        *dnoise_dx += temp2 * x2;
        *dnoise_dy += temp2 * y2;
        *dnoise_dz += temp2 * z2;
        *dnoise_dw += temp2 * w2;
        float temp3 = t23 * t3 * (g3[0] * x3 + g3[1] * y3 + g3[2] * z3 + g3[3] * w3);
        *dnoise_dx += temp3 * x3;
        *dnoise_dy += temp3 * y3;
        *dnoise_dz += temp3 * z3;
        *dnoise_dw += temp3 * w3;
        float temp4 = t24 * t4 * (g4[0] * x4 + g4[1] * y4 + g4[2] * z4 + g4[3] * w4);
        *dnoise_dx += temp4 * x4;
        *dnoise_dy += temp4 * y4;
        *dnoise_dz += temp4 * z4;
        *dnoise_dw += temp4 * w4;
        *dnoise_dx *= -8.0f;
        *dnoise_dy *= -8.0f;
        *dnoise_dz *= -8.0f;
        *dnoise_dw *= -8.0f;
        *dnoise_dx += t40 * g0[0] + t41 * g1[0] + t42 * g2[0] + t43 * g3[0] + t44 * g4[0];
        *dnoise_dy += t40 * g0[1] + t41 * g1[1] + t42 * g2[1] + t43 * g3[1] + t44 * g4[1];
        *dnoise_dz += t40 * g0[2] + t41 * g1[2] + t42 * g2[2] + t43 * g3[2] + t44 * g4[2];
        *dnoise_dw += t40 * g0[3] + t41 * g1[3] + t42 * g2[3] + t43 * g3[3] + t44 * g4[3];
        // Scale derivative to match the noise scaling
        *dnoise_dx *= scale;
        *dnoise_dy *= scale;
        *dnoise_dz *= scale;
        *dnoise_dw *= scale;
      }

    return noise;
}



} // namespace pvt
OSL_NAMESPACE_EXIT

