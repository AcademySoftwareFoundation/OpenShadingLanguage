// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

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

shader __OSL_CONCAT(test_texture3d_opts_, __OSL_XMACRO_SUFFIX) (
      output vector out_alpha = 0,
      output vector out_alpha_derivs = 0,
      output vector out_blur = 0,
      output vector out_color = 0,
      output vector out_dx = 0,
      output vector out_dy = 0,
      output vector out_errormsg = 0,
      output vector out_firstchannel = 0,
      output vector out_missingalpha = 0,
      output vector out_missing_color = 0,
      output vector out_simple = 0,
      output vector out_smallderivs = 0,
      output vector out_width = 0,
      output vector out_widthderivs = 0,
      output vector out_wrap = 0,
      )
{

#if defined(VARYING_FILENAME)
    string filename = "data/sphere.vdb";
    if (v > 0.85)
        filename = "missing.tx";
#elif defined(UNIFORM_FILENAME)
    string filename = "data/sphere.vdb";
    if (raytype("camera") == 0)
        filename = "data/sphere.vdb";
#elif defined(CONSTANT_FILENAME)
    string filename = "data/sphere.vdb";
#endif

    point vP = point(u*2-1,v*2-1,0);

    {
        // This tests single-channel reads as well as "alpha" (one past the
        // last channel directly read, stored in another variable)
        float r, g, b;
        r = (float) texture3d (filename, vP);
        int firstchannel_val = init(1);
        g = (float) texture3d (filename, vP, "firstchannel", firstchannel_val, "alpha", b);
        out_alpha = color (r, g, b);
    }

    {
        float a=-1.0;
        string wrap_val = init("clamp");
        color C = texture3d (filename, vP, "alpha", a, "wrap", wrap_val);
        out_alpha_derivs = color (a, Dx(a)/Dx(u), Dy(a)/Dy(v));
    }

    {
        float b = init(pow (u/2, 2.0), 0.5);
        out_blur = (color) texture3d (filename, vP, "blur", b);
    }

    {
        out_color = (color) texture3d (filename, vP);
        out_dx = Dx (out_color) * 128.0;
        out_dy = Dy (out_color) * 128.0;
    }

    {
        string err = init("uninitialized");
        string filename = (u > 0.5) ? "bad.tif" : filename;
        color C = (color) texture3d (filename, vP, "errormessage", err);
        if (err == "") {
            out_errormsg = mix (color(0,1,0), C, 0.75);
        } else {
            out_errormsg = color(1,0,0);
            if (err != "unknown")
                printf ("err %0.03g : %s\n", vP, err);
        }
    }

    {
        int firstchannel_val = init(1);
        float fill_val = init(0.5);
        out_firstchannel = (color) texture3d (filename, vP, "firstchannel", firstchannel_val, "fill", fill_val);
    }
    
    {
        float alpha = 0;
        float missingalpha_val = init(float(int(u*8+v*8))/16.0,0.75);
        float x = texture3d (filename, vP, "missingalpha", missingalpha_val,
                "alpha", alpha);
        out_missingalpha = alpha;
        if (alpha != 0.75)
            error ("missingalpha did not work\n");
    }
    
    {
        color missingcolor_val = init(color(1,0,0));
        out_missing_color = (color) texture3d (filename, vP, "missingcolor", missingcolor_val);
    }
    
    {
        out_simple = (color) texture3d (filename, vP);
    }
    
    {
        float xwidth = u * 2e-8;
        float ywidth = v * 2e-8;
        float zwidth = (2.0-(u+v)) * 2e-8;
        float blur_val = init(0.01);
        out_smallderivs = (color) texture3d (filename, vP, vector(xwidth, 0, 0), vector(0, ywidth, 0), vector(0, 0, zwidth), "blur", blur_val);
    }

    // TODO: add test of "time" attribute.

    {
        float width_val = init(1+u*10, 0.5);
        out_width = (color) texture3d (filename, vP, "width", width_val);
    }

    {
        float xwidth = u*u * 10;
        float ywidth = v*v * 5;
        out_widthderivs = (color) texture3d (filename, vP, Dx(vP)*xwidth, Dy(vP)*ywidth, vector(0,0,0));
    }

    {
        string wrap_varying = "default";
        if (u > 0.2) wrap_varying = "black";
        if (u > 0.4) wrap_varying = "periodic";
        if (u > 0.6) wrap_varying = "clamp";
        if (u > 0.6) wrap_varying = "mirror";
        string wrap_val = init(wrap_varying,"clamp");
        out_wrap = (color) texture3d (filename, point(-0.1 + 2 * u, -0.2 + 2 * v, -0.3 + (u+v)),
                "wrap", wrap_val);
    }
}
