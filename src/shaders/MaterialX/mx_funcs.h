// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage
//
// MaterialX specification (c) 2017 Lucasfilm Ltd.
// http://www.materialx.org/

#pragma once
#include "color2.h"
#include "color4.h"
#include "vector2.h"
#include "vector4.h"
#include "matrix33.h"


///////////////////////////////////////////////////////////////////////////
// This file contains lots of functions helpful in the implementation of
// the MaterialX nodes.
///////////////////////////////////////////////////////////////////////////


// Define mx_convert_type
//   float -> colvecN
color mx_convert (float a) { return color(a); }
color2 mx_convert (float a) { return color2(a,a); }
color4 mx_convert (float a) { return color4(a,a); }
vector mx_convert (float a) { return vector(a); }
vector2 mx_convert (float a) { return vector2(a,a); }
vector4 mx_convert (float a) { return vector4(a,a,a,a); }
//   colN <-> vecN
vector mx_convert (color a) { return (vector)a; }
vector2 mx_convert (color2 a) { return vector2 (a.r, a.a); }
vector4 mx_convert (color4 a) { return vector4 (a.rgb[0], a.rgb[1], a.rgb[2], a.a); }
color mx_convert (vector a) { return (color)a; }
color2 mx_convert (vector2 a) { return color2 (a.x, a.y); }
color4 mx_convert (vector4 a) { return color4 ({a.x,a.y,a.z}, a.w); }
//   col3 <-> col4
color mx_convert (color4 a) { return a.rgb; }
color4 mx_convert (color a) { return color4(a,1.0); }

// Define mx_add() overloaded for all MX types.
float mx_add (float a, float b) { return a+b; }
point mx_add (point a, point b) { return a+b; }
point mx_add (point a, float b) { return a+b; }
vector mx_add (vector a, vector b) { return a+b; }
vector mx_add (vector a, float b) { return a+b; }
vector2 mx_add (vector2 a, vector2 b) { return a+b; }
vector2 mx_add (vector2 a, float b) { return a+b; }
vector4 mx_add (vector4 a, vector4 b) { return a+b; }
vector4 mx_add (vector4 a, float b) { return a+b; }
color mx_add (color a, color b) { return a+b; }
color mx_add (color a, float b) { return a+b; }
color2 mx_add (color2 a, color2 b) { return a+b; }
color2 mx_add (color2 a, float b) { return a+b; }
color4 mx_add (color4 a, color4 b) { return a+b; }
color4 mx_add (color4 a, float b) { return a+b; }
closure color mx_add (closure color a, closure color b) { return a+b; }

matrix33 mx_add(matrix33 a, matrix33 b)
{
    return matrix33 (matrix(
        a.m[0][0]+b.m[0][0], a.m[0][1]+b.m[0][1], a.m[0][2]+b.m[0][2], 0.0,
        a.m[1][0]+b.m[1][0], a.m[1][1]+b.m[1][1], a.m[1][2]+b.m[1][2], 0.0,
        a.m[2][0]+b.m[2][0], a.m[2][1]+b.m[2][1], a.m[2][2]+b.m[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0));
}

matrix33 mx_add(matrix33 a, float b)
{
    return matrix33 (matrix(a.m[0][0]+b, a.m[0][1]+b, a.m[0][2]+b, 0.0,
                            a.m[1][0]+b, a.m[1][1]+b, a.m[1][2]+b, 0.0,
                            a.m[2][0]+b, a.m[2][1]+b, a.m[2][2]+b, 0.0,
                            0.0, 0.0, 0.0, 1.0));
}

matrix mx_add(matrix a, matrix b)
{
    return matrix (a[0][0]+b[0][0], a[0][1]+b[0][1], a[0][2]+b[0][2], a[0][3]+b[0][3],
                   a[1][0]+b[1][0], a[1][1]+b[1][1], a[1][2]+b[1][2], a[1][3]+b[1][3],
                   a[2][0]+b[2][0], a[2][1]+b[2][1], a[2][2]+b[2][2], a[2][3]+b[2][3],
                   a[3][0]+b[3][0], a[3][1]+b[3][1], a[3][2]+b[3][2], a[3][3]+b[3][3]);
}

matrix mx_add(matrix a, float b)
{
    return matrix (a[0][0]+b, a[0][1]+b, a[0][2]+b, a[0][3]+b,
                   a[1][0]+b, a[1][1]+b, a[1][2]+b, a[1][3]+b,
                   a[2][0]+b, a[2][1]+b, a[2][2]+b, a[2][3]+b,
                   a[3][0]+b, a[3][1]+b, a[3][2]+b, a[3][3]+b);
}


// Define mx_sub() overloaded for all MX types.
float mx_sub (float a, float b) { return a-b; }
point mx_sub (point a, point b) { return a-b; }
point mx_sub (point a, float b) { return a-b; }
vector mx_sub (vector a, vector b) { return a-b; }
vector mx_sub (vector a, float b) { return a-b; }
vector2 mx_sub (vector2 a, vector2 b) { return a-b; }
vector2 mx_sub (vector2 a, float b) { return a-b; }
vector4 mx_sub (vector4 a, vector4 b) { return a-b; }
vector4 mx_sub (vector4 a, float b) { return a-b; }
color mx_sub (color a, color b) { return a-b; }
color mx_sub (color a, float b) { return a-b; }
color2 mx_sub (color2 a, color2 b) { return a-b; }
color2 mx_sub (color2 a, float b) { return a-b; }
color4 mx_sub (color4 a, color4 b) { return a-b; }
color4 mx_sub (color4 a, float b) { return a-b; }

matrix33 mx_sub (matrix33 a, matrix33 b)
{
    return matrix33 (matrix(
        a.m[0][0]-b.m[0][0], a.m[0][1]-b.m[0][1], a.m[0][2]-b.m[0][2], 0.0,
        a.m[1][0]-b.m[1][0], a.m[1][1]-b.m[1][1], a.m[1][2]-b.m[1][2], 0.0,
        a.m[2][0]-b.m[2][0], a.m[2][1]-b.m[2][1], a.m[2][2]-b.m[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0));
}

matrix33 mx_sub (matrix33 a, float b)
{
    return matrix33 (matrix(
        a.m[0][0]-b, a.m[0][1]-b, a.m[0][2]-b, 0.0,
        a.m[1][0]-b, a.m[1][1]-b, a.m[1][2]-b, 0.0,
        a.m[2][0]-b, a.m[2][1]-b, a.m[2][2]-b, 0.0,
        0.0, 0.0, 0.0, 1.0));
}

matrix mx_sub (matrix a, matrix b)
{
    return matrix(a[0][0]-b[0][0], a[0][1]-b[0][1], a[0][2]-b[0][2], a[0][3]-b[0][3],
                  a[1][0]-b[1][0], a[1][1]-b[1][1], a[1][2]-b[1][2], a[1][3]-b[1][3],
                  a[2][0]-b[2][0], a[2][1]-b[2][1], a[2][2]-b[2][2], a[2][3]-b[2][3],
                  a[3][0]-b[3][0], a[3][1]-b[3][1], a[3][2]-b[3][2], a[3][3]-b[3][3]);
}

matrix mx_sub (matrix a, float b)
{
    return matrix (a[0][0]-b, a[0][1]-b, a[0][2]-b, a[0][3]-b,
                   a[1][0]-b, a[1][1]-b, a[1][2]-b, a[1][3]-b,
                   a[2][0]-b, a[2][1]-b, a[2][2]-b, a[2][3]-b,
                   a[3][0]-b, a[3][1]-b, a[3][2]-b, a[3][3]-b);
}



// remap `in` from [inLow, inHigh] to [outLow, outHigh], optionally clamping
// to the new range.
//
float remap(float in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      float x = (in - inLow)/(inHigh-inLow);
      if (doClamp == 1) {
           x = clamp(x, 0, 1);
      }
      return outLow + (outHigh - outLow) * x;
}

color remap(color in, color inLow, color inHigh, color outLow, color outHigh, int doClamp)
{
      color x = (in - inLow) / (inHigh - inLow);
      if (doClamp == 1) {
           x = clamp(x, 0, 1);
      }
      return outLow + (outHigh - outLow) * x;
}

color remap(color in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      color x = (in - inLow) / (inHigh - inLow);
      if (doClamp == 1) {
           x = clamp(x, 0, 1);
      }
      return outLow + (outHigh - outLow) * x;
}

color2 remap(color2 c, color2 inLow, color2 inHigh, color2 outLow, color2 outHigh, int doClamp)
{
      return color2(remap(c.r, inLow.r, inHigh.r, outLow.r, outHigh.r, doClamp),
                    remap(c.a, inLow.a, inHigh.a, outLow.a, outHigh.a, doClamp));
}

color2 remap(color2 c, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    return remap(c, color2(inLow, inLow), color2(inHigh, inHigh), color2(outLow, outLow), color2(outHigh, outHigh), doClamp);
}

color4 remap(color4 c, color4 inLow, color4 inHigh, color4 outLow, color4 outHigh, int doClamp)
{
      return color4(remap(c.rgb, inLow.rgb, inHigh.rgb, outLow.rgb, outHigh.rgb, doClamp),
                    remap(c.a, inLow.a, inHigh.a, outLow.a, outHigh.a, doClamp));
}

color4 remap(color4 c, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    color4 c4_inLow = color4(color(inLow), inLow);
    color4 c4_inHigh = color4(color(inHigh), inHigh);
    color4 c4_outLow = color4(color(outLow), outLow);
    color4 c4_outHigh = color4(color(outHigh), outHigh);
    return remap(c, c4_inLow, c4_inHigh, c4_outLow, c4_outHigh, doClamp);
}

vector2 remap(vector2 in, vector2 inLow, vector2 inHigh, vector2 outLow, vector2 outHigh, int doClamp)
{
    return vector2 (remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp),
                    remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp));
}

vector2 remap(vector2 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    return vector2 (remap(in.x, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.y, inLow, inHigh, outLow, outHigh, doClamp));
}

vector4 remap(vector4 in, vector4 inLow, vector4 inHigh, vector4 outLow, vector4 outHigh, int doClamp)
{
    return vector4 (remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp),
                    remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp),
                    remap(in.z, inLow.z, inHigh.z, outLow.z, outHigh.z, doClamp),
                    remap(in.w, inLow.w, inHigh.w, outLow.w, outHigh.w, doClamp));
}

vector4 remap(vector4 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    return vector4 (remap(in.x, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.y, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.z, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.w, inLow, inHigh, outLow, outHigh, doClamp));
}



float fgamma(float in, float g)
{
    return sign(in) * pow(abs(in), g);
}

color fgamma(color in, color g)
{
    return sign(in) * pow(abs(in), g);
}

color fgamma(color in, float g)
{
    return sign(in) * pow(abs(in), g);
}

color2 fgamma(color2 c, color2 a)
{
    return color2(fgamma(c.r, a.r), fgamma(c.a, a.a));
}

color2 fgamma(color2 c, float a)
{
    return fgamma(c, color2(a, a));
}

color4 fgamma(color4 a, color4 b)
{
    return color4(fgamma(a.rgb, b.rgb), fgamma(a.a, b.a));
}

color4 fgamma(color4 a, float b)
{
    return fgamma(a, color4(color(b), b));
}

vector2 fgamma(vector2 in, vector2 g)
{
    return vector2 (fgamma(in.x, g.x), fgamma(in.y, g.y));
}

vector2 fgamma(vector2 in, float g)
{
    return vector2 (fgamma(in.x, g), fgamma(in.y, g));
}

vector4 fgamma(vector4 in, vector4 g)
{
    return vector4 (fgamma(in.x, g.x),
                    fgamma(in.y, g.y),
                    fgamma(in.z, g.z),
                    fgamma(in.w, g.w));
}

vector4 fgamma(vector4 in, float g)
{
    return vector4 (fgamma(in.x, g),
                    fgamma(in.y, g),
                    fgamma(in.z, g),
                    fgamma(in.w, g));
}



//
// contrast scales the input around a central `pivot` value.
//
float contrast(float in, float amount, float pivot)
{
    float out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color contrast(color in, color amount, color pivot)
{
    color out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color contrast(color in, float amount, float pivot)
{
    color out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color2 contrast(color2 c, color2 amount, color2 pivot)
{
    return color2(contrast(c.r, amount.r, pivot.r),
                  contrast(c.a, amount.a, pivot.a));
}

color2 contrast(color2 c, float amount, float pivot)
{
    return contrast(c, color2(amount, amount), color2(pivot, pivot));
}

color4 contrast(color4 c, color4 amount, color4 pivot)
{
    return color4(contrast(c.rgb, amount.rgb, pivot.rgb),
                  contrast(c.a, amount.a, pivot.a));
}

color4 contrast(color4 c, float amount, float pivot)
{
    return contrast(c, color4(color(amount), amount), color4(color(pivot), pivot));
}

vector2 contrast(vector2 in, vector2 amount, vector2 pivot)
{
    return vector2 (contrast(in.x, amount.x, pivot.x),
                    contrast(in.y, amount.y, pivot.y));
}

vector2 contrast(vector2 in, float amount, float pivot)
{
    return contrast(in, vector2(amount, amount), vector2(pivot, pivot));
}

vector4 contrast(vector4 in, vector4 amount, vector4 pivot)
{
    return vector4 (contrast(in.x, amount.x, pivot.x),
                    contrast(in.y, amount.y, pivot.y),
                    contrast(in.z, amount.z, pivot.z),
                    contrast(in.w, amount.w, pivot.w));
}

vector4 contrast(vector4 in, float amount, float pivot)
{
    return vector4 (contrast(in.x, amount, pivot),
                    contrast(in.y, amount, pivot),
                    contrast(in.z, amount, pivot),
                    contrast(in.w, amount, pivot));
}



color2 noise (string noisetype, float x, float y)
{
    color cnoise = (color) noise (noisetype, x, y);
    return color2 (cnoise[0], cnoise[1]);
}

vector2 noise (string noisetype, float x, float y)
{
    color cnoise = (color) noise (noisetype, x, y);
    return vector2 (cnoise[0], cnoise[1]);
}

color4 noise (string noisetype, float x, float y)
{
    color cnoise = (color) noise (noisetype, x, y);
    float fnoise = (float) noise (noisetype, x + 19, y + 73);
    return color4 (cnoise, fnoise);
}

vector4 noise (string noisetype, float x, float y)
{
    color cnoise = (color) noise (noisetype, x, y);
    float fnoise = (float) noise (noisetype, x + 19, y + 73);
    return vector4 (cnoise[0], cnoise[1], cnoise[2], fnoise);
}


color2 noise (string noisetype, point position)
{
    color cnoise = (color) noise (noisetype, position);
    return color2 (cnoise[0], cnoise[1]);
}

vector2 noise (string noisetype, point position)
{
    color cnoise = (color) noise (noisetype, position);
    return vector2 (cnoise[0], cnoise[1]);
}

color4 noise (string noisetype, point position)
{
    color cnoise = (color) noise (noisetype, position);
    float fnoise = (float) noise (noisetype, position+vector(19,73,29));
    return color4 (cnoise, fnoise);
}

vector4 noise (string noisetype, point position)
{
    color cnoise = (color) noise (noisetype, position);
    float fnoise = (float) noise (noisetype, position+vector(19,73,29));
    return vector4 (cnoise[0], cnoise[1], cnoise[2], fnoise);
}



color2 cellnoise (float x, float y)
{
    color cnoise = (color) cellnoise (x, y);
    return color2 (cnoise[0], cnoise[1]);
}

vector2 cellnoise (float x, float y)
{
    color cnoise = (color) cellnoise (x, y);
    return vector2 (cnoise[0], cnoise[1]);
}

color4 cellnoise (float x, float y)
{
    color cnoise = (color) cellnoise (x, y);
    float fnoise = (float) cellnoise (x + 19, y + 73);
    return color4 (cnoise, fnoise);
}

vector4 cellnoise (float x, float y)
{
    color cnoise = (color) cellnoise (x, y);
    float fnoise = (float) cellnoise (x + 19, y + 73);
    return vector4 (cnoise[0], cnoise[1], cnoise[2], fnoise);
}



color2 cellnoise (point position)
{
    color cnoise = (color) cellnoise (position);
    return color2 (cnoise[0], cnoise[1]);
}

vector2 cellnoise (point position)
{
    color cnoise = (color) cellnoise (position);
    return vector2 (cnoise[0], cnoise[1]);
}

color4 cellnoise (point position)
{
    color cnoise = (color) cellnoise (position);
    float fnoise = (float) cellnoise (position+vector(19,73,29));
    return color4 (cnoise, fnoise);
}

vector4 cellnoise (point position)
{
    color cnoise = (color) cellnoise (position);
    float fnoise = (float) cellnoise (position+vector(19,73,29));
    return vector4 (cnoise[0], cnoise[1], cnoise[2], fnoise);
}



//
// fractional Brownian motion
//
float fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{
    float out = 0;
    float amp = 1.0;
    point p = position;

    for (int i = 0;  i < octaves;  i += 1) {
        out += amp * noise(noisetype, p);
        amp *= diminish;
        p *= lacunarity;
    }
    return out;
}

color fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{
    color out = 0;
    float amp = 1.0;
    point p = position;

    for (int i = 0;  i < octaves;  i += 1) {
        out += amp * (color)noise(noisetype, p);
        amp *= diminish;
        p *= lacunarity;
    }
    return out;
}

color2 fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{
    return color2 ((float) fBm (position, octaves, lacunarity, diminish, noisetype),
                   (float) fBm (position+point(19, 193, 17), octaves, lacunarity, diminish, noisetype));
}

vector2 fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{
    return vector2 ((float) fBm (position, octaves, lacunarity, diminish, noisetype),
                    (float) fBm (position+point(19, 193, 17), octaves, lacunarity, diminish, noisetype));
}

color4 fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{
    color c = (color) fBm (position, octaves, lacunarity, diminish, noisetype);
    float f = (float) fBm (position+point(19, 193, 17), octaves, lacunarity, diminish, noisetype);
    return color4 (c, f);
}

vector4 fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{
    color c = (color) fBm (position, octaves, lacunarity, diminish, noisetype);
    float f = (float) fBm (position+point(19, 193, 17), octaves, lacunarity, diminish, noisetype);
    return vector4 (c[0], c[1], c[2], f);
}






float swizzle_float (float in[4], string channels)
{
    float out;
    float outF[4];
    int c_len = strlen(channels);

    for (int i=0; i<c_len; i++) {
        string ch = substr(channels, i, 1);
        if (ch == "r" || ch == "x")
            outF[i] = in[0];
        else if (ch == "g" || ch == "y")
            outF[i] = in[1];
        else if (ch == "b" || ch == "z")
            outF[i] = in[2];
        else if (ch == "a" || ch == "w")
            outF[i] = in[3];
        else if(ch == "1")
            outF[i] = 1;
        else
            outF[i] = 0;
    }
    out = outF[0];
    return out;
}



color swizzle_color (float in[4], string channels)
{
    color out;
    float outF[4];
    int c_len = strlen(channels);

    for (int i=0; i<c_len; i++) {
        string ch = substr(channels, i, 1);
        if (ch == "r" || ch == "x")
            outF[i] = in[0];
        else if (ch == "g" || ch == "y")
            outF[i] = in[1];
        else if (ch == "b" || ch == "z")
            outF[i] = in[2];
        else if (ch == "a" || ch == "w")
            outF[i] = in[3];
        else if(ch == "1")
            outF[i] = 1;
        else
            outF[i] = 0;
    }
    return color(outF[0],outF[1],outF[2]);
}



vector swizzle_vector (float in[4], string channels)
{
    vector out;
    float outF[4];
    int c_len = strlen(channels);

    for (int i=0; i<c_len; i++) {
        string ch = substr(channels, i, 1);
        if (ch == "r" || ch == "x")
            outF[i] = in[0];
        else if (ch == "g" || ch == "y")
            outF[i] = in[1];
        else if (ch == "b" || ch == "z")
            outF[i] = in[2];
        else if (ch == "a" || ch == "w")
            outF[i] = in[3];
        else if(ch == "1")
            outF[i] = 1;
        else
            outF[i] = 0;
    }
    return vector(outF[0],outF[1],outF[2]);
}



color2 swizzle_color2 (float in[4], string channels)
{
    color2  out;
    float outF[4];
    int c_len = strlen(channels);

    for (int i=0; i<c_len; i++) {
        string ch = substr(channels, i, 1);
        if (ch == "r" || ch == "x")
            outF[i] = in[0];
        else if (ch == "g" || ch == "y")
            outF[i] = in[1];
        else if (ch == "b" || ch == "z")
            outF[i] = in[2];
        else if (ch == "a")
            outF[i] = in[1];
        else if(ch == "1")
            outF[i] = 1;
        else
            outF[i] = 0;
    }
    out.r = outF[0];
    out.a = outF[1];

    return out;
}



color4 swizzle_color4 (float in[4], string channels)
{
    color4  out;
    float outF[4];
    int c_len = strlen(channels);

    for (int i=0; i<c_len; i++) {
        string ch = substr(channels, i, 1);
        if (ch == "r" || ch == "x")
            outF[i] = in[0];
        else if (ch == "g" || ch == "y")
            outF[i] = in[1];
        else if (ch == "b" || ch == "z")
            outF[i] = in[2];
        else if (ch == "a" || ch == "w")
            outF[i] = in[3];
        else if(ch == "1")
            outF[i] = 1;
        else
            outF[i] = 0;
    }
    out.rgb = color(outF[0],outF[1],outF[2]);
    out.a = outF[3];

    return out;
}


vector2 swizzle_vector2 (float in[4], string channels)
{
    vector2  out;
    float outF[4];
    int c_len = strlen(channels);

    for (int i=0; i<c_len; i++) {
        string ch = substr(channels, i, 1);
        if (ch == "r" || ch == "x")
            outF[i] = in[0];
        else if (ch == "g" || ch == "y")
            outF[i] = in[1];
        else if (ch == "b" || ch == "z")
            outF[i] = in[2];
        else if (ch == "a" || ch == "w")
            outF[i] = in[3];
        else if(ch == "1")
            outF[i] = 1;
        else
            outF[i] = 0;
    }
    out.x = outF[0];
    out.y = outF[1];

    return out;
}



vector4 swizzle_vector4 (float in[4], string channels)
{
    vector4  out;
    float outF[4];
    int c_len = strlen(channels);

    for (int i=0; i<c_len; i++) {
        string ch = substr(channels, i, 1);
        if (ch == "r" || ch == "x")
            outF[i] = in[0];
        else if (ch == "g" || ch == "y")
            outF[i] = in[1];
        else if (ch == "b" || ch == "z")
            outF[i] = in[2];
        else if (ch == "a" || ch == "w")
            outF[i] = in[3];
        else if(ch == "1")
            outF[i] = 1;
        else
            outF[i] = 0;
    }
    out.x = outF[0];
    out.y = outF[1];
    out.z = outF[2];
    out.w = outF[3];
    return out;
}


//
// setup_missing_color_alpha() implements all the type permutations for
// setting up missingColor and missingAlpha given the default_value (and its
// specific type).
//

void setup_missing_color_alpha (float default_value,
              output color missingColor, output float missingAlpha)
{
    missingColor = default_value;
    missingAlpha = 1;
}

void setup_missing_color_alpha (color default_value,
              output color missingColor, output float missingAlpha)
{
    missingColor = default_value;
    missingAlpha = 1;
}


void setup_missing_color_alpha (vector default_value,
              output color missingColor, output float missingAlpha)
{
    missingColor = vector (default_value);
    missingAlpha = 1;
}

void setup_missing_color_alpha (color2 default_value,
              output color missingColor, output float missingAlpha)
{
    missingColor = color (default_value.r, default_value.a, 0);
    missingAlpha = 1;
}

void setup_missing_color_alpha (vector2 default_value,
              output color missingColor, output float missingAlpha)
{
    missingColor = color (default_value.x, default_value.y, 0);
    missingAlpha = 1;
}

void setup_missing_color_alpha (vector4 default_value,
              output color missingColor, output float missingAlpha)
{
    missingColor = color (default_value.x, default_value.y, default_value.z);
    missingAlpha = default_value.w;
}

void setup_missing_color_alpha (color4 default_value,
              output color missingColor, output float missingAlpha)
{
    missingColor = color (default_value.rgb);
    missingAlpha = default_value.a;
}



//
// combine() combines an up to 4 floats, or an rgb and alpha, into the given
// return type, in a way that makes as much sense as possible.
//
float combine (float a, float b, float c, float d)
{
    return a;
}

color combine (float a, float b, float c, float d)
{
    return color (a, b, c);
}

vector combine (float a, float b, float c, float d)
{
    return vector (a, b, c);
}

color2 combine (float a, float b, float c, float d)
{
    return color2 (a, b);
}

vector2 combine (float a, float b, float c, float d)
{
    return vector2 (a, b);
}

color4 combine (float a, float b, float c, float d)
{
    return color4 (color(a,b,c), d);
}

vector4 combine (float a, float b, float c, float d)
{
    return vector4 (a, b, c, d);
}


float combine (color rgb, float alpha)
{
    return rgb[0];
}

color combine (color rgb, float alpha)
{
    return rgb;
}

vector combine (color rgb, float alpha)
{
    return (vector)rgb;
}

color2 combine (color rgb, float alpha)
{
    return color2 (rgb[0], rgb[1]);
}

vector2 combine (color rgb, float alpha)
{
    return vector2 (rgb[0], rgb[1]);
}

color4 combine (color rgb, float alpha)
{
    return color4 (rgb, alpha);
}

vector4 combine (color rgb, float alpha)
{
    return vector4 (rgb[0], rgb[1], rgb[2], alpha);
}


//
// extract(in,index) returns one indexed float from the aggregate.
//

float extract (color2 in, int index)
{
    return index == 0 ? in.r : in.a;
}

float extract (vector2 in, int index)
{
    return index == 0 ? in.x : in.y;
}


float extract (color in, int index)
{
    return in[index];
}

float extract (vector in, int index)
{
    return in[index];
}


float extract (color4 in, int index)
{
    return index < 3 ? in.rgb[index] : in.a;
}

float extract (vector4 in, int index)
{
    float r;
    if      (index == 0) r = in.x;
    else if (index == 2) r = in.y;
    else if (index == 3) r = in.z;
    else                 r = in.w;
    return r;
}



// DEPRECATED: MatrialX <= 1.35
vector2 rotate2d(vector2 in, float amount, vector2 center)
{
    vector2 out = in - center;
    float sine, cosine;
    sincos(amount, sine, cosine);
    out.x = in.x * cosine - in.y * sine;
    out.y = in.y * cosine + in.x * sine;
    out = out + center;
    return out;
}

vector2 rotate (vector2 in, float amount,
                vector axis /*unused in the 2D case*/)
{
    vector2 out = in;
    float sine, cosine;
    sincos(amount, sine, cosine);
    out.x = in.x * cosine - in.y * sine;
    out.y = in.y * cosine + in.x * sine;
    out = out;
    return out;
}
