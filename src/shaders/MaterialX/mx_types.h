// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once

#ifdef FLOAT
#define TYPE float
#define TYPE_STR "Float"
#define TYPE_ZERO 0
#define TYPE_ZERO_POINT_FIVE 0.5
#define TYPE_ONE 1
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "r"
#define SHADER_NAME(NAME)  NAME ## _float

#elif COLOR
#define TYPE color
#define TYPE_STR "Color"
#define TYPE_ZERO 0
#define TYPE_ZERO_POINT_FIVE 0.5
#define TYPE_ONE 1
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "rgb"
#define SHADER_NAME(NAME)  NAME ## _color

#elif VECTOR
#define TYPE vector
#define TYPE_STR "Vector"
#define TYPE_ZERO 0
#define TYPE_ZERO_POINT_FIVE 0.5
#define TYPE_ONE 1
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "xyz"
#define SHADER_NAME(NAME)  NAME ## _vector

#elif COLOR2
#define TYPE color2
#define TYPE_STR "Color2"
#define TYPE_ZERO {0, 0}
#define TYPE_ZERO_POINT_FIVE {0.5, 0.5}
#define TYPE_ONE {1, 1}
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "rg"
#define SHADER_NAME(NAME)  NAME ## _color2

#elif VECTOR2
#define TYPE vector2
#define TYPE_STR "Vector2"
#define TYPE_ZERO {0, 0}
#define TYPE_ZERO_POINT_FIVE {0.5, 0.5}
#define TYPE_ONE {1, 1}
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "xy"
#define SHADER_NAME(NAME)  NAME ## _vector2

#elif COLOR4
#define TYPE color4
#define TYPE_STR "Color4"
#define TYPE_ZERO {color(0, 0, 0), 0}
#define TYPE_ZERO_POINT_FIVE {color(0.5, 0.5, 0.5), 0.5}
#define TYPE_ONE {color(1, 1, 1), 1}
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "rgba"
#define SHADER_NAME(NAME)  NAME ## _color4

#elif VECTOR4
#define TYPE vector4
#define TYPE_STR "Vector4"
#define TYPE_ZERO {0, 0, 0, 0}
#define TYPE_ZERO_POINT_FIVE {0.5, 0.5, 0.5, 0.5}
#define TYPE_ONE {1, 1, 1, 1}
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "xyzw"
#define SHADER_NAME(NAME)  NAME ## _vector4

#elif MATRIX44
#define TYPE matrix
#define TYPE_STR "matrix44"
#define TYPE_ZERO matrix(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)
#define TYPE_ZERO_POINT_FIVE  matrix(0.5,0,0,0, 0,0.5,0,0, 0,0,0.5,0, 0,0,0,0.5)
#define TYPE_ONE  matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "xyzw"
#define SHADER_NAME(NAME)  NAME ## _matrix44

#elif MATRIX33
#define TYPE matrix
#define TYPE_STR "matrix33"
#define TYPE_ZERO  matrix(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)
#define TYPE_ZERO_POINT_FIVE  matrix(0.5,0,0,0, 0,0.5,0,0, 0,0,0.5,0, 0,0,0,0)
#define TYPE_ONE  matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,0)
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "xyzw"
#define SHADER_NAME(NAME)  NAME ## _matrix33

#elif BOOL
#define TYPE int
#define TYPE_STR "bool"
#define TYPE_ZERO 0
#define TYPE_ZERO_POINT_FIVE 1
#define TYPE_ONE 1
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "x"
#define SHADER_NAME(NAME)  NAME ## _bool

#elif INT
#define TYPE int
#define TYPE_STR "int"
#define TYPE_ZERO 0
#define TYPE_ZERO_POINT_FIVE 1
#define TYPE_ONE 1
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "x"
#define SHADER_NAME(NAME)  NAME ## _int

#elif STRING
#define TYPE string
#define TYPE_STR "string"
#define TYPE_ZERO "zero"
#define TYPE_ZERO_POINT_FIVE "zero point five"
#define TYPE_ONE "one"
#define TYPE_DEFAULT_IN "default"
#define TYPE_DEFAULT_OUT "default"
#define TYPE_DEFAULT_CHANNELS "a"
#define SHADER_NAME(NAME)  NAME ## _string

#elif FILENAME
#define TYPE string
#define TYPE_STR "filename"
#define TYPE_ZERO "zero"
#define TYPE_ZERO_POINT_FIVE "zero point five"
#define TYPE_ONE "one"
#define TYPE_DEFAULT_IN "default"
#define TYPE_DEFAULT_OUT "default"
#define TYPE_DEFAULT_CHANNELS "a"
#define SHADER_NAME(NAME)  NAME ## _filename

#elif SURFACESHADER
#define TYPE closure color
#define TYPE_STR "surfaceshader"
#define TYPE_ZERO 0
#define TYPE_ZERO_POINT_FIVE 0
#define TYPE_ONE 1
#define TYPE_DEFAULT_IN TYPE_ZERO
#define TYPE_DEFAULT_OUT TYPE_ZERO
#define TYPE_DEFAULT_CHANNELS "a"
#define SHADER_NAME(NAME)  NAME ## _surfaceshader
#endif

#ifndef TYPE
#error Invalid or no Type specified in compile flags
#endif

#include "vector2.h"
#include "vector4.h"
#include "color2.h"
#include "color4.h"
#include "mx_funcs.h"

