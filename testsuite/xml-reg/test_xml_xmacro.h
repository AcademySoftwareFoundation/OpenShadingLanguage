// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "oslutil.h"

#ifndef __OSL_XMACRO_SUFFIX
#    error must define __OSL_XMACRO_SUFFIX to create a unique testname before including this header
#endif

#if !defined(VARYING_FILENAME) && !defined(UNIFORM_FILENAME) && !defined(CONSTANT_FILENAME)
#    error Must define either VARYING_FILENAME, UNIFORM_FILENAME, CONSTANT_FILENAME before including this xmacro!
#endif 

#if !defined(VARYING_DATA) && !defined(UNIFORM_DATA) && !defined(CONSTANT_DATA)
#    error Must define either VARYING_DATA, UNIFORM_DATA, CONSTANT_DATA before including this xmacro!
#endif

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)


#ifdef CONSTANT_DATA
// leave existing constant val undisturbed
int init(int val) {
    return val;
}
float init(float val) {
    return val;
}
float init(float val, float const_init) {
    return const_init;
}
color init(color val) {
    return val;
}
string init(string val) {
    return val;
}
string init(string val, string const_init) {
    return const_init;
}
#endif

#ifdef UNIFORM_DATA

// raytype is uniform so the resulting string
// should be uniform and not a constant
int init(int val) {
    // Intended to be unreachable, but
    // prevent constant folding of result
    if (raytype("camera") == 0)
        return -1;
    return val;
}
float init(float val) {
    // Intended to be unreachable, but
    // prevent constant folding of result
    if (raytype("camera") == 0)
        return -1.0;
    return val;
}
float init(float val, float const_init) {
    // Intended to be unreachable, but
    // prevent constant folding of result
    if (raytype("camera") == 0)
        return -1.0;
    return const_init;
}
color init(color val) {
    // Intended to be unreachable, but
    // prevent constant folding of result
    if (raytype("camera") == 0)
        return color(-1);
    return val;
}
string init(string val) {
    // Intended to be unreachable, but
    // prevent constant folding of result
    if (raytype("camera") == 0)
        return "unreachable";
    return val;
}
string init(string val, string const_init) {
    // Intended to be unreachable, but
    // prevent constant folding of result
    if (raytype("camera") == 0)
        return "unreachable";
    return const_init;
}

#endif

#ifdef VARYING_DATA
// 'u' and 'v' are varying so val
// should be forced to be varying

int init(int val) {
    // Intended to be unreachable, but
    // force result to be varying
    if (u < -100000.0)
        return int(v);
    return val;
}
float init(float val) {
    // Intended to be unreachable, but
    // force result to be varying
    if (u < -100000.0)
        return v;
    return val;
}
float init(float val, float const_init) {
    // Intended to be unreachable, but
    // force result to be varying
    if (u < -100000.0)
        return v;
    return val;
}
color init(color val) {
    // Intended to be unreachable, but
    // force result to be varying
    if (u < -100000.0)
        return color(v,u, v+u);
    return val;
}
string init(string val) {
    // Intended to be unreachable, but
    // force result to be varying
    if (u < -100000.0)
        return "unreachable";
    return val;
}
string init(string val, string const_init) {
    // Intended to be unreachable, but
    // force result to be varying
    if (u < -100000.0)
        return "unreachable";
    return val;
}
#endif

float visualizeNodeId(int nodeId) {
    return (1 + clamp(nodeId, -1,1))*0.5;
}

shader __OSL_CONCAT(test_xml_, __OSL_XMACRO_SUFFIX) (
      output color out_nomatchId = 0,
      output color out_camerapackId = 0,
      output color out_imageId = 0,
      output color out_cameraId = 0,
      output color out_nocameraId = 0,
      output color out_foundName = 0,
      output color out_name = 0,
      output color out_found2sides = 0,
      output color out_2sides = 0,
      output color out_xformId = 0,
      output color out_foundMat = 0,
      output color out_mat = 0,
      output color out_foundChannel = 0,
      output color out_channel = 0,
      output color out_foundFilter = 0,
      output color out_filter = 0,
      )
{

#if defined(VARYING_FILENAME)
    string filename = "test.xml";
    if (v > 0.45)
        filename = "test2.xml";
    if (v > 0.70)
        filename = "test3.xml";
#elif defined(UNIFORM_FILENAME)
    string filename = "test.xml";
    if (raytype("camera") == 0)
        filename = "test3.xml";
#elif defined(CONSTANT_FILENAME)
    string filename = "test.xml";
#endif

    int nomatchId = dict_find (filename, init("//nomatch"));

    out_nomatchId = color(visualizeNodeId(nomatchId));

    int camerapackId1 = dict_find (filename, init("//camerapack"));
    int camerapackId2 = dict_next (camerapackId1);
    int camerapackId3 = dict_next (camerapackId2);

    out_camerapackId = color(visualizeNodeId(camerapackId1),
                                visualizeNodeId(camerapackId2),
                                visualizeNodeId(camerapackId3));

    int imageId1 = dict_find (filename, init("//image"));
    int imageId2 = dict_next (imageId1);
    int imageId3 = dict_next (imageId2);
    int imageId4 = dict_next (imageId3);
    
    out_imageId = color(visualizeNodeId(imageId1),
                                (visualizeNodeId(imageId2) + visualizeNodeId(imageId3))/4,
                                visualizeNodeId(imageId4));

    int nomatchId2 = dict_find (camerapackId1, init("nomatch"));

    int cameraId1 = dict_find (camerapackId1, init("camera"));
    int cameraId2 = dict_find (camerapackId2, init("camera"));
    int cameraId3 = dict_find (camerapackId3, init("camera"));

    out_cameraId = color(visualizeNodeId(cameraId1),
                         visualizeNodeId(cameraId2),
                         visualizeNodeId(cameraId3));

    int nocameraId1 = dict_find (imageId1, init("camera"));
    int nocameraId2 = dict_find (imageId2, init("camera"));
    int nocameraId3 = dict_find (imageId3, init("camera"));

    out_nocameraId = color(visualizeNodeId(nocameraId1),
                       visualizeNodeId(nocameraId2),
                       visualizeNodeId(nocameraId3));
    
    string name1 = init("error");
    string name2 = init("error");
    string name3 = init("error");
    int foundName1 = dict_value (cameraId1, init("name"), name1);
    int foundName2 = dict_value (cameraId2, init("name"), name2);
    int foundName3 = dict_value (cameraId3, init("name"), name3);

    out_foundName = color(visualizeNodeId(foundName1),
                          visualizeNodeId(foundName2),
                          visualizeNodeId(foundName3));

    out_name = mix (color(0,0,0), color(1,1,1),
                draw_string(name1, u*150, v*150, 1, 1, 0));

    out_name = mix (out_name, color(1,1,1),
                draw_string(name2, 75 + u*150, 75 + v*150, 1, 1, 0));
    out_name = mix (out_name, color(1,1,1),
                draw_string(name3, 150 + u*150, 150 + v*150, 1, 1, 0));

    int twosides1 = init(0);
    int twosides2 = init(0);
    int twosides3 = init(0);
    int found2sides1 = dict_value (camerapackId1, init("twoSidesOn"), twosides1);
    int found2sides2 = dict_value (camerapackId2, init("twoSidesOn"), twosides2);
    int found2sides3 = dict_value (camerapackId3, init("twoSidesOn"), twosides3);

    out_found2sides = color(visualizeNodeId(found2sides1),
                          visualizeNodeId(found2sides2),
                          visualizeNodeId(found2sides3));

    out_2sides = color(visualizeNodeId(twosides1),
                       visualizeNodeId(twosides2),
                       visualizeNodeId(twosides3));

    int xformId1 = dict_find (cameraId1, init("xform"));
    int xformId2 = dict_find (cameraId2, init("xform"));
    int xformId3 = dict_find (cameraId3, init("xform"));

    out_xformId = color(visualizeNodeId(xformId1),
                         visualizeNodeId(xformId2),
                         visualizeNodeId(xformId3));

    matrix mat1 = matrix(init(u,0));
    matrix mat2 = matrix(init(v,0));
    matrix mat3 = matrix(init(u+v,0));
    int foundMat1 = dict_value (xformId1, init("matrix"), mat1);
    int foundMat2 = dict_value (xformId2, init("matrix"), mat2);
    int foundMat3 = dict_value (xformId3, init("matrix"), mat3);

    out_foundMat = color(visualizeNodeId(foundMat1),
                         visualizeNodeId(foundMat2),
                         visualizeNodeId(foundMat3));

    out_mat = (transform(mat1,P) + transform(mat2,P) + transform(mat3,P))/3.0;


    string channel1 = init("error");
    string channel2 = init("error");
    string channel3 = init("error");
    string channel4 = init("error");
    int foundChannel1 = dict_value (imageId1, init("channel"), channel1);
    int foundChannel2 = dict_value (imageId2, init("channel"), channel2);
    int foundChannel3 = dict_value (imageId3, init("channel"), channel3);
    int foundChannel4 = dict_value (imageId4, init("channel"), channel4);

    out_foundChannel = color(visualizeNodeId(foundChannel1),
                          visualizeNodeId(foundChannel2 + foundChannel3),
                          visualizeNodeId(foundChannel4));

    if (u < 0.5) {
    out_channel = mix (color(0,0,0), color(1,1,1),
                draw_string(channel1, u*200, v*200, 1, 1, 0));
    out_channel = mix (out_channel, color(1,1,1),
                draw_string(channel2, 100 + u*200, 100 + v*200, 1, 1, 0));
    } else {
        out_channel = mix (color(0,0,0), color(1,1,1),
                    draw_string(channel3, u*200, v*200, 1, 1, 0));
        out_channel = mix (out_channel, color(1,1,1),
                    draw_string(channel4, 100 + u*200, 100 + v*200, 1, 1, 0));
    }


    int filterId1 = dict_find (cameraId1, init("filter"));
    int filterId2 = dict_find (cameraId2, init("filter"));
    int filterId3 = dict_find (cameraId3, init("filter"));
    out_foundFilter = color(visualizeNodeId(filterId1),
                          visualizeNodeId(filterId2),
                          visualizeNodeId(filterId3));


    color filter1 = color(1);
    color filter2 = color(1);
    color filter3 = color(1);
    int foundfilter1 = dict_value (filterId1, init("color"), filter1);
    int foundfilter2 = dict_value (filterId2, init("color"), filter2);
    int foundfilter3 = dict_value (filterId3, init("color"), filter3);

    out_foundFilter = color(visualizeNodeId(foundfilter1),
                          visualizeNodeId(foundfilter2),
                          visualizeNodeId(foundfilter3));

    out_filter = filter1;
    if (u > 0.1) out_filter = filter2;
    if (u > 0.9) out_filter = filter3;

    // OSLC will complain if an array is used with dict_value,
    // whether that is legal/correct is under investigation
#if 0
    float filterArray1[3];
    float filterArray2[3];
    float filterArray3[3];
    int foundfilterArray1 = dict_value (filterId1, init("color"), filterArray1);
    int foundfilterArray2 = dict_value (filterId2, init("color"), filterArray2);
    int foundfilterArray3 = dict_value (filterId3, init("color"), filterArray3);

    out_foundFilter = color(visualizeNodeId(foundfilterArray1),
                          visualizeNodeId(foundfilterArray2),
                          visualizeNodeId(foundfilterArray3));

    out_filter = color(filterArray1[0],filterArray1[1],filterArray1[2]);
    if (u > 0.1) out_filter = color(filterArray2[0],filterArray2[1],filterArray2[2]);
    if (u > 0.9) out_filter = color(filterArray3[0],filterArray3[1],filterArray3[2]);
#endif
}
