// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_XMACRO_SUFFIX
#    error must define __OSL_XMACRO_SUFFIX to create a unique testname before including this header
#endif

#if !defined(VARYING_FILENAME) && !defined(UNIFORM_FILENAME) && !defined(CONSTANT_FILENAME)
#    error Must define either VARYING_FILENAME, UNIFORM_FILENAME, CONSTANT_FILENAME before including this xmacro!
#endif 

#if !defined(VARYING_PARAM_NAME) && !defined(UNIFORM_PARAM_NAME) && !defined(CONSTANT_PARAM_NAME)
#    error Must define either VARYING_PARAM_NAME, UNIFORM_PARAM_NAME, CONSTANT_PARAM_NAME before including this xmacro!
#endif

#if !defined(VARYING_DATA) && !defined(UNIFORM_DATA) && !defined(CONSTANT_DATA)
#    error Must define either VARYING_DATA, UNIFORM_DATA, CONSTANT_DATA before including this xmacro!
#endif

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)

#ifdef CONSTANT_PARAM_NAME
string param_name(string name) {
    return name;
}
#endif

#ifdef UNIFORM_PARAM_NAME
string param_name(string name) {
    // raytype is uniform so the resulting string
    // should be uniform and not a constant
    if (raytype("camera") == 0)
        return "unreachable";
    return name;
}
#endif

#ifdef VARYING_PARAM_NAME
string param_name(string name) {
    // 'u' is varying so the resulting string
    // should be varying and not a constant
    if (u < -10000000.0)
        return "unreachable";
    return name;
}
#endif

#ifdef CONSTANT_DATA
// leave existing constant val undisturbed
void init(int val) {
}
void init(float val) {
}
void init(color val) {
}
void init(string val) {
}
void init(int val[2]) {
}
void init(int val[4]) {
}
void init(matrix val) {
}
#endif

#ifdef UNIFORM_DATA

// raytype is uniform so the resulting string
// should be uniform and not a constant
void init(int val) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0)
        val = -1;
}
void init(float val) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0)
        val = -1.0;
}
void init(color val) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0)
        val = color(-1.0,-1.0,-1.0);
}
void init(string val) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0)
        val = "unreachable";
}
void init(int val[2]) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0) {
        val[0] = -1.0;
        val[1] = -1.0;
    }
}
void init(int val[4]) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0) {
        val[0] = -1.0;
        val[1] = -1.0;
        val[2] = -1.0;
        val[3] = -1.0;
    }
}
void init(matrix val) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0) {
        val[0][0] = -1.0;
        val[1][1] = -1.0;
        val[2][2] = -1.0;
    }
}

#endif

#ifdef VARYING_DATA
// 'u' and 'v' are varying so val
// should be forced to be varying

void init(int val) {
    // Intended to be unreachable, but
    // force val to be varying
    if (u < -100000.0)
        val = int(v);
}
void init(float val) {
    // Intended to be unreachable, but
    // force val to be varying
    if (u < -100000.0)
        val = v;
}
void init(color val) {
    // Intended to be unreachable, but
    // force val to be varying
    if (u < -100000.0)
        val = color(v,v,v);
}
void init(string val) {
    // Intended to be unreachable, but
    // force val to be varying
    if (u < -100000.0)
        val = "unreachable";
}
void init(int val[2]) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0) {
        val[0] = int(v);
        val[1] = int(v);
    }
}
void init(int val[4]) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0) {
        val[0] = int(v);
        val[1] = int(v);
        val[2] = int(v);
        val[3] = int(v);
    }
}
void init(matrix val) {
    // Intended to be unreachable, but
    // prevent constant folding of val
    if (raytype("camera") == 0) {
        val[0][0] = int(v);
        val[1][1] = int(v);
        val[2][2] = int(v);
    }
}
#endif

shader __OSL_CONCAT(test_gettextureinfo_, __OSL_XMACRO_SUFFIX) (
      output vector out_resolution = 0,
      output vector out_channels = 0,
      output vector out_texturetype = 0,
      output vector out_textureformat = 0,
      output vector out_datawin = 0,
      output vector out_dispwin = 0,
      output vector out_worldtocamera = 0,
      output vector out_worldtoscreen = 0,
      output vector out_datetime = 0,
      output vector out_avgcolor = 0,
      output vector out_avgalpha = 0,
      output vector out_constcolor = 0,
      output vector out_constalpha = 0,
      output vector out_unfoundinfo = 0,
      output vector out_unfoundfile = 0,
      output vector out_skipcondition = 0,
      output vector out_exists = 0,
      output vector out_not_exists = 0,
      )
{
    int r = 0;


#if defined(VARYING_FILENAME)
    string filename = "../common/textures/grid.tx";
    if (v > 0.33)
        filename = "../common/textures/mandrill.tif"; 
    if (v > 0.66)
        filename = "../common/textures/kitchen_probe.hdr"; 
    if (v > 0.8)
        filename = "../common/textures/nan.exr"; 
#elif defined(UNIFORM_FILENAME)
    string filename = "../common/textures/grid.tx";
    if (raytype("camera") == 0)
        filename = "../common/textures/mandrill.tif";
#elif defined(CONSTANT_FILENAME)
    string filename = "../common/textures/grid.tx";
#endif

    int resolution[2];
    resolution[0] = 4;
    resolution[1] = 5;
    init(resolution);
    
    r = gettextureinfo (filename, param_name("resolution"), resolution);
    out_resolution = vector(1.0, 1.0, 1.0);
    if (r) {
        out_resolution = vector(sin(u/0.1*resolution[0]), tan(u/1.3*resolution[1]),tan(u-v/1.2*r));
    }

    int channels;
    init(channels);
    r = gettextureinfo (filename, param_name("channels"), channels);
    out_channels = vector(1.0, 1.0, 1.0);
    if (r) {
        out_channels = vector(sin(u/0.1*channels), tan(u/1.3*r),tan(u-v/1.2*r));
    }

    string texturetype = "unknown";
    init(texturetype);
    r = gettextureinfo (filename, param_name("texturetype"), texturetype);
    out_texturetype = vector(1.0, 1.0, 1.0);
    if (r) {
        if (texturetype == "Plain Texture")
            out_texturetype = vector(0.9, 0.2, 0.1);
        else if (texturetype == "Shadow")
            out_texturetype = vector(0.2, 0.9, 0.1);
        else if (texturetype == "Environment")
            out_texturetype = vector(0.1, 0.2, 0.9);
        else if (texturetype == "Volume")
            out_texturetype = vector(0.8, 0.1, 0.9);
        else
            out_texturetype = vector(0.0, 0.0, 0.0);
    }

    string textureformat = "unknown";
    init(textureformat);
    r = gettextureinfo (filename, param_name("textureformat"), textureformat);
    out_textureformat = vector(1.0, 1.0, 1.0);
    if (r) {
        if (textureformat == "Plain Texture")
            out_textureformat = vector(0.9, 0.2, 0.1);
        else if (textureformat == "Shadow")
            out_textureformat = vector(0.2, 0.9, 0.1);
        else if (textureformat == "CubeFace Shadow")
            out_textureformat = vector(0.1, 0.2, 0.9);
        else if (textureformat == "Volume Shadow")
            out_textureformat = vector(0.3, 0.9, 0.9);
        else if (textureformat == "CubeFace Environment")
            out_textureformat = vector(0.9, 0.4, 0.6);
        else if (textureformat == "LatLong Environment")
            out_textureformat = vector(0.4, 0.2, 0.6);
        else if (textureformat == "Volume Texture")
            out_textureformat = vector(0.8, 0.1, 0.9);
        else
            out_textureformat = vector(0.0, 0.0, 0.0);
    }

    int datawin[4];
    datawin[0] = 1;
    datawin[1] = 8;
    datawin[2] = 4;
    datawin[3] = 3;
    init(datawin);
    r = gettextureinfo (filename, param_name("datawindow"), datawin);
    out_datawin = vector(1.0, 1.0, 1.0);
    if (r) {
        out_datawin = vector(u/0.1*datawin[0]+datawin[1], u/1.3*r,u-v/1.2*datawin[2]+datawin[3]);
    }
            
            
    int dispwin[4];
    dispwin[0] = 1;
    dispwin[1] = 8;
    dispwin[2] = 4;
    dispwin[3] = 3;
    init(dispwin);
    r = gettextureinfo (filename, param_name("displaywindow"), dispwin);
    out_dispwin = vector(1.0, 1.0, 1.0);
    if (r) {
        out_dispwin = vector(u/0.1*dispwin[0]+dispwin[1], (u/1.3)*r, u-v/1.2*dispwin[2]+dispwin[3]);
    }

    matrix worldtocamera = 0;
    init(worldtocamera);
    r = gettextureinfo (filename, param_name("worldtocamera"), worldtocamera);
    out_worldtocamera = vector(0.0);
    if (r) {
        out_worldtocamera = vector(
                worldtocamera[0][0] + worldtocamera[0][1] + worldtocamera[0][2] + worldtocamera[0][3] + worldtocamera[3][0],
                worldtocamera[1][0] + worldtocamera[1][1] + worldtocamera[1][2] + worldtocamera[1][3] + worldtocamera[3][1],
                worldtocamera[2][0] + worldtocamera[2][1] + worldtocamera[2][2] + worldtocamera[2][3] + worldtocamera[3][2]);
    }

    matrix worldtoscreen = 0;
    init(worldtoscreen);
    r = gettextureinfo (filename, param_name("worldtoscreen"), worldtoscreen);
    out_worldtocamera = vector(0.0);
    if (r) {
        out_worldtocamera = vector(
                worldtoscreen[0][0] + worldtoscreen[0][1] + worldtoscreen[0][2] + worldtoscreen[0][3] + worldtoscreen[3][0],
                worldtoscreen[1][0] + worldtoscreen[1][1] + worldtoscreen[1][2] + worldtoscreen[1][3] + worldtoscreen[3][1],
                worldtoscreen[2][0] + worldtoscreen[2][1] + worldtoscreen[2][2] + worldtoscreen[2][3] + worldtoscreen[3][2]);
    }

    // Test arbitrary metadata
    string datetime;
    init(datetime);
    r = gettextureinfo (filename, param_name("DateTime"), datetime);
    out_datetime = vector(1.0, 1.0, 1.0);
    if (r) {
        out_datetime = vector(0.5, 0.5, 0.5);
    }

    // Test average and constant retrieval
    color avgcolor;
    init(avgcolor);
    r = gettextureinfo (filename, param_name("averagecolor"), avgcolor);
    out_avgcolor = vector(1.0, 1.0, 1.0);
    if (r) {
        out_avgcolor = avgcolor;
    }
    
    float avgalpha;
    init(avgalpha);
    r = gettextureinfo (filename, param_name("averagealpha"), avgalpha);
    out_avgalpha = vector(1.0, 1.0, 1.0);
    if (r) {
        out_avgalpha = vector(avgalpha);
    }
    
    color constcolor;
    init(constcolor);
    r = gettextureinfo (filename, param_name("constantcolor"), constcolor);
    out_constcolor = vector(1.0, 1.0, 1.0);
    if (r) {
        out_constcolor = constcolor;
    }
    
    float constalpha;
    init(constalpha);
    r = gettextureinfo (filename, param_name("constantalpha"), constalpha);
    out_constalpha = vector(1.0, 1.0, 1.0);
    if (r) {
        out_constalpha = vector(constalpha);
    }

    // Test failure of unfound info name
    string foobar = "not found";
    init(foobar);
    r = gettextureinfo (filename, param_name("foobar"), foobar);
    out_unfoundinfo = vector(1.0, 1.0, 1.0);
    if (foobar == "not found") {
        out_unfoundinfo = vector(0.5, 0.5, 0.5);
    }
    if (r) {
        out_unfoundinfo = vector(0.0, 0.0, 0.0);
    }

    // Test failure of unfound file name
    string data = "not found";
    string badfile = "badfile";
    init(data);
    r = gettextureinfo (badfile, param_name("textureformat"), data);
    out_unfoundfile = vector(1.0, 1.0, 1.0);
    if (data == "not found") {
        out_unfoundfile = vector(0.5, 0.5, 0.5);
    }
    if (r) {
        out_unfoundfile = vector(0.0, 0.0, 0.0);
    }

    // Make a query of a bad file inside a conditional -- the idea is that
    // we should NOT see an error message if the statement is not executed.
    out_skipcondition = vector(1.0, 1.0, 1.0);
    if (u > 2) {
        string data2 = "not found";
        init(data2);
        string badfile2 = "badfile2";
        r = gettextureinfo (badfile2, param_name("textureformat"), data2);
        out_skipcondition = vector(0.0, 0.0, 0.0);
    }
    // Test existence of a valid file
    {
        int e;
        init(e);
        r = gettextureinfo (filename, param_name("exists"), e);
        out_exists = vector(0.5, 0.5, 0.5);
        if (r) {
            out_exists = vector(e,e,e);
        }
    }

    // Test existence of a nonexistant file
    {
        int e;
        init(e);
        string badfile3 = "badfile3";
        r = gettextureinfo (badfile3, param_name("exists"), e);
        out_not_exists = vector(0.5, 0.5, 0.5);
        if (r) {
            out_not_exists = vector(e,e,e);
        }
    }
}
