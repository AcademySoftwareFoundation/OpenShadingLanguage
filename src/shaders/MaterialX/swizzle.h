// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once
#include "mx_types.h"

// TODO: refine logic to not over compute for loop

TYPE swizzle(float in[4], string expression){
    
    TYPE  out;
    float outF[4]; 
    int i = 0;
    string channels = expression;
    int c_len = strlen(channels);
    
    for(i=0; i<c_len; i++)
    {
        string ch = substr(channels, i, 1);
        
        if(ch == "r" || ch == "x"){
            outF[i] = in[0];
        }
        else if(ch == "g" || ch == "y"){
            outF[i] = in[1];
        }
        else if(ch == "b" || ch == "z"){
            outF[i] = in[2];
        }
        else if(ch == "a" || ch == "w"){
            outF[i] = in[3];
        }
        else if(ch == "1" ){
            outF[i] = 1;
        }
        else {
            outF[i] = 0;
        }
    }
    #if defined(FLOAT)
        out = outF[0];
    #elif defined(COLOR)
        out = color(outF[0],outF[1],outF[2]);
    #elif defined(VECTOR)
        out = vector(outF[0],outF[1],outF[2]);
    #elif defined(COLOR2) 
        out.r = outF[0];
        out.g = outF[1];
    #elif defined(COLOR4)
        out.rgb = color(outF[0],outF[1],outF[2]);
        out.a = outF[3];
    #elif defined(VECTOR2)
        out.x = outF[0];
        out.y = outF[1];
    #elif defined(VECTOR4)
        out.x = outF[0];
        out.y = outF[1];
        out.z =    outF[2];
        out.w = outF[3];
    #endif
    
    return out;
}



