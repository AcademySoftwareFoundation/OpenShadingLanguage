/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagebufalgo_util.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/timer.h>

#include "OSL/oslexec.h"
#include "OSL/oslcomp.h"
#include "OSL/oslquery.h"
#include "simplerend.h"
using namespace OSL;
using OIIO::TypeDesc;
using OIIO::ParamValue;
using OIIO::ParamValueList;

static ShadingSystem *shadingsys = NULL;
static std::vector<std::string> shadernames;
static std::vector<std::string> outputfiles;
static std::vector<std::string> outputvars;
static std::vector<ustring> outputvarnames;
static std::vector<OIIO::ImageBuf*> outputimgs;
static std::string dataformatname = "";
static std::string shaderpath;
static std::vector<std::string> entrylayers;
static std::vector<std::string> entryoutputs;
static std::vector<int> entrylayer_index;
static std::vector<const ShaderSymbol *> entrylayer_symbols;
static bool debug = false;
static bool debug2 = false;
static bool verbose = false;
static bool runstats = false;
static int profile = 0;
static bool O0 = false, O1 = false, O2 = false;
static bool pixelcenters = false;
static bool debugnan = false;
static bool debug_uninit = false;
static bool use_group_outputs = false;
static bool do_oslquery = false;
static bool inbuffer = false;
static bool use_shade_image = false;
static bool userdata_isconnected = false;
static int xres = 1, yres = 1;
static int num_threads = 0;
static std::string groupname;
static std::string groupspec;
static std::string layername;
static std::vector<std::string> connections;
static ParamValueList params;
static ParamValueList reparams;
static std::string reparam_layer;
static ErrorHandler errhandler;
static int iters = 1;
static std::string raytype = "camera";
static int raytype_bit = 0;
static bool raytype_opt = false;
static std::string extraoptions;
static SimpleRenderer rend;  // RendererServices
static OSL::Matrix44 Mshad;  // "shader" space to "common" space matrix
static OSL::Matrix44 Mobj;   // "object" space to "common" space matrix
static ShaderGroupRef shadergroup;
static std::string archivegroup;
static int exprcount = 0;
static bool shadingsys_options_set = false;
static float sscale = 1, tscale = 1;
static float soffset = 0, toffset = 0;

#ifndef OSL_USE_WIDE_LLVM_BACKEND
	#error OSL_USE_WIDE_LLVM_BACKEND must be defined [0|1]
#endif

static void
inject_params ()
{
    for (size_t p = 0;  p < params.size();  ++p) {
        const ParamValue &pv (params[p]);
        shadingsys->Parameter (pv.name(), pv.type(), pv.data(),
                               pv.interp() == ParamValue::INTERP_CONSTANT);
    }
}



static void
set_shadingsys_options ()
{
    if (shadingsys_options_set)
        return;
    shadingsys->attribute ("llvm_debug", 2);
    shadingsys->attribute ("debug", debug2 ? 2 : (debug ? 1 : 0));
    shadingsys->attribute ("compile_report", debug|debug2);
    int opt = 2;  // default
    if (O0) opt = 0;
    if (O1) opt = 1;
    if (O2) opt = 2;
    if (const char *opt_env = getenv ("TESTSHADE_OPT"))  // overrides opt
        opt = atoi(opt_env);
    shadingsys->attribute ("optimize", opt);
    shadingsys->attribute ("lockgeom", 1);
    shadingsys->attribute ("debug_nan", debugnan);
    shadingsys->attribute ("debug_uninit", debug_uninit);
    shadingsys->attribute ("userdata_isconnected", userdata_isconnected);
    if (! shaderpath.empty())
        shadingsys->attribute ("searchpath:shader", shaderpath);
    
#if OSL_USE_WIDE_LLVM_BACKEND
    // For batched operations turn off coalesce temps
    shadingsys->attribute ("opt_coalesce_temps", 0);
#endif
    
    shadingsys_options_set = true;
}



/// Read the entire contents of the named file and place it in str,
/// returning true on success, false on failure.
inline bool
read_text_file (string_view filename, std::string &str)
{
    std::ifstream in (filename.c_str(), std::ios::in | std::ios::binary);
    if (in) {
        std::ostringstream contents;
        contents << in.rdbuf();
        in.close ();
        str = contents.str();
        return true;
    }
    return false;
}



static void
compile_buffer (const std::string &sourcecode,
                const std::string &shadername)
{
    // std::cout << "source was\n---\n" << sourcecode << "---\n\n";
    std::string osobuffer;
    OSLCompiler compiler;
    std::vector<std::string> options;

    if (! compiler.compile_buffer (sourcecode, osobuffer, options)) {
        std::cerr << "Could not compile \"" << shadername << "\"\n";
        exit (EXIT_FAILURE);
    }
    // std::cout << "Compiled to oso:\n---\n" << osobuffer << "---\n\n";

    if (! shadingsys->LoadMemoryCompiledShader (shadername, osobuffer)) {
        std::cerr << "Could not load compiled buffer from \""
                  << shadername << "\"\n";
        exit (EXIT_FAILURE);
    }
}



static void
shader_from_buffers (std::string shadername)
{
    std::string oslfilename = shadername;
    if (! OIIO::Strutil::ends_with (oslfilename, ".osl"))
        oslfilename += ".osl";
    std::string sourcecode;
    if (! read_text_file (oslfilename, sourcecode)) {
        std::cerr << "Could not open \"" << oslfilename << "\"\n";
        exit (EXIT_FAILURE);
    }

    compile_buffer (sourcecode, shadername);
    // std::cout << "Read and compiled " << shadername << "\n";
}



static int
add_shader (int argc, const char *argv[])
{
    ASSERT (argc == 1);
    string_view shadername (argv[0]);
    std::cout << "SHADER NAME=" << shadername << std::endl;

    set_shadingsys_options ();

    if (inbuffer)  // Request to exercise the buffer-based API calls
        shader_from_buffers (shadername);

    for (int i = 0;  i < argc;  i++) {
        inject_params ();
        shadernames.push_back (shadername);
        shadingsys->Shader ("surface", shadername, layername);
        layername.clear ();
        params.clear ();
    }
    return 0;
}



// The --expr ARG command line option will take ARG that is a snipped of
// OSL source code, embed it in some boilerplate shader wrapper, compile
// it from memory, and run that in the same way that would have been done
// if it were a compiled shader on disk. The boilerplate assumes that there
// are two output parameters for the shader: color result, and float alpha.
//
// Example use:
//   testshade -v -g 64 64 -o result out.exr -expr 'result=color(u,v,0);'
//
static void
specify_expr (int argc, const char *argv[])
{
    ASSERT (argc == 2);
    std::string shadername = OIIO::Strutil::format("expr_%d", exprcount++);
    std::string sourcecode =
        "shader " + shadername + " (\n"
        "    float s = u [[ int lockgeom=0 ]],\n"
        "    float t = v [[ int lockgeom=0 ]],\n"
        "    output color result = 0,\n"
        "    output float alpha = 1,\n"
        "  )\n"
        "{\n"
        "    " + std::string(argv[1]) + "\n"
        "    ;\n"
        "}\n";
    if (verbose)
        std::cout << "Expression-based shader text is:\n---\n"
                  << sourcecode << "---\n";

    set_shadingsys_options ();

    compile_buffer (sourcecode, shadername);

    inject_params ();
    shadernames.push_back (shadername);
    shadingsys->Shader ("surface", shadername, layername);
    layername.clear ();
    params.clear ();
}



static void
action_param (int argc, const char *argv[])
{
    std::string command = argv[0];
    bool use_reparam = false;
    if (OIIO::Strutil::istarts_with(command, "--reparam") ||
        OIIO::Strutil::istarts_with(command, "-reparam"))
        use_reparam = true;
    ParamValueList &params (use_reparam ? reparams : (::params));

    string_view paramname (argv[1]);
    string_view stringval (argv[2]);
    TypeDesc type = TypeDesc::UNKNOWN;
    bool unlockgeom = false;
    float f[16];

    size_t pos;
    while ((pos = command.find_first_of(":")) != std::string::npos) {
        command = command.substr (pos+1, std::string::npos);
        std::vector<std::string> splits;
        OIIO::Strutil::split (command, splits, ":", 1);
        if (splits.size() < 1) {}
        else if (OIIO::Strutil::istarts_with(splits[0],"type="))
            type.fromstring (splits[0].c_str()+5);
        else if (OIIO::Strutil::istarts_with(splits[0],"lockgeom="))
            unlockgeom = (strtol (splits[0].c_str()+9, NULL, 10) == 0);
    }

    // If it is or might be a matrix, look for 16 comma-separated floats
    if ((type == TypeDesc::UNKNOWN || type == TypeDesc::TypeMatrix)
        && sscanf (stringval.c_str(),
                   "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                   &f[0], &f[1], &f[2], &f[3],
                   &f[4], &f[5], &f[6], &f[7], &f[8], &f[9], &f[10], &f[11],
                   &f[12], &f[13], &f[14], &f[15]) == 16) {
        params.push_back (ParamValue());
        params.back().init (paramname, TypeDesc::TypeMatrix, 1, f);
        if (unlockgeom)
            params.back().interp (ParamValue::INTERP_VERTEX);
        return;
    }
    // If it is or might be a vector type, look for 3 comma-separated floats
    if ((type == TypeDesc::UNKNOWN || equivalent(type,TypeDesc::TypeVector))
        && sscanf (stringval.c_str(), "%g, %g, %g", &f[0], &f[1], &f[2]) == 3) {
        if (type == TypeDesc::UNKNOWN)
            type = TypeDesc::TypeVector;
        params.push_back (ParamValue());
        params.back().init (paramname, type, 1, f);
        if (unlockgeom)
            params.back().interp (ParamValue::INTERP_VERTEX);
        return;
    }
    // If it is or might be an int, look for an int that takes up the whole
    // string.
    if ((type == TypeDesc::UNKNOWN || type == TypeDesc::TypeInt)) {
        char *endptr = NULL;
        int ival = strtol(stringval.c_str(),&endptr,10);
        if (endptr && *endptr == 0) {
            params.push_back (ParamValue());
            params.back().init (paramname, TypeDesc::TypeInt, 1, &ival);
            if (unlockgeom)
                params.back().interp (ParamValue::INTERP_VERTEX);
            return;
        }
    }
    // If it is or might be an float, look for a float that takes up the
    // whole string.
    if ((type == TypeDesc::UNKNOWN || type == TypeDesc::TypeFloat)) {
        char *endptr = NULL;
        float fval = (float) strtod(stringval.c_str(),&endptr);
        if (endptr && *endptr == 0) {
            params.push_back (ParamValue());
            params.back().init (paramname, TypeDesc::TypeFloat, 1, &fval);
            if (unlockgeom)
                params.back().interp (ParamValue::INTERP_VERTEX);
            return;
        }
    }

    // Catch-all for float types and arrays
    if (type.basetype == TypeDesc::FLOAT) {
        int n = type.aggregate * type.numelements();
        std::vector<float> vals (n);
        for (int i = 0;  i < n;  ++i) {
            OIIO::Strutil::parse_float (stringval, vals[i]);
            OIIO::Strutil::parse_char (stringval, ',');
        }
        params.push_back (ParamValue());
        params.back().init (paramname, type, 1, &vals[0]);
        if (unlockgeom)
            params.back().interp (ParamValue::INTERP_VERTEX);
        return;
    }

    // Catch-all for int types and arrays
    if (type.basetype == TypeDesc::INT) {
        int n = type.aggregate * type.numelements();
        std::vector<int> vals (n);
        for (int i = 0;  i < n;  ++i) {
            OIIO::Strutil::parse_int (stringval, vals[i]);
            OIIO::Strutil::parse_char (stringval, ',');
        }
        params.push_back (ParamValue());
        params.back().init (paramname, type, 1, &vals[0]);
        if (unlockgeom)
            params.back().interp (ParamValue::INTERP_VERTEX);
        return;
    }

    // All remaining cases -- it's a string
    const char *s = stringval.c_str();
    params.push_back (ParamValue());
    params.back().init (paramname, TypeDesc::TypeString, 1, &s);
    if (unlockgeom)
        params.back().interp (ParamValue::INTERP_VERTEX);
}



// reparam -- just set reparam_layer and then let action_param do all the
// hard work.
static void
action_reparam (int argc, const char *argv[])
{
    reparam_layer = argv[1];
    const char *newargv[] = { argv[0], argv[2], argv[3] };
    action_param (3, newargv);
}



static void
action_groupspec (int argc, const char *argv[])
{
    shadingsys->ShaderGroupEnd ();
    std::string groupspec (argv[1]);
    if (OIIO::Filesystem::exists (groupspec)) {
        // If it names a file, use the contents of the file as the group
        // specification.
        OIIO::Filesystem::read_text_file (groupspec, groupspec);
        set_shadingsys_options ();
    }
    if (verbose)
        std::cout << "Processing group specification:\n---\n"
                  << groupspec << "\n---\n";
    shadergroup = shadingsys->ShaderGroupBegin (groupname, "surface", groupspec);
}



static void
set_profile (int argc, const char *argv[])
{
    profile = 1;
    shadingsys->attribute ("profile", profile);
}



static void
getargs (int argc, const char *argv[])
{
    static bool help = false;
    OIIO::ArgParse ap;
    ap.options ("Usage:  testshade [options] shader...",
                "%*", add_shader, "",
                "--help", &help, "Print help message",
                "-v", &verbose, "Verbose messages",
                "-t %d", &num_threads, "Render using N threads (default: auto-detect)",
                "--debug", &debug, "Lots of debugging info",
                "--debug2", &debug2, "Even more debugging info",
                "--runstats", &runstats, "Print run statistics",
                "--stats", &runstats, "",  // DEPRECATED 1.7
                "--profile %@", &set_profile, NULL, "Print profile information",
                "--path %s", &shaderpath, "Specify oso search path",
                "-g %d %d", &xres, &yres, "Make an X x Y grid of shading points",
                "-res %d %d", &xres, &yres, "", // synonym for -g
                "--options %s", &extraoptions, "Set extra OSL options",
                "-o %L %L", &outputvars, &outputfiles,
                        "Output (variable, filename)",
                "-od %s", &dataformatname, "Set the output data format to one of: "
                        "uint8, half, float",
                "--groupname %s", &groupname, "Set shader group name",
                "--layer %s", &layername, "Set next layer name",
                "--param %@ %s %s", &action_param, NULL, NULL,
                        "Add a parameter (args: name value) (options: type=%s, lockgeom=%d)",
                "--connect %L %L %L %L",
                    &connections, &connections, &connections, &connections,
                    "Connect fromlayer fromoutput tolayer toinput",
                "--reparam %@ %s %s %s", &action_reparam, NULL, NULL, NULL,
                        "Change a parameter (args: layername paramname value) (options: type=%s)",
                "--group %@ %s", &action_groupspec, &groupspec,
                        "Specify a full group command",
                "--archivegroup %s", &archivegroup,
                        "Archive the group to a given filename",
                "--raytype %s", &raytype, "Set the raytype",
                "--raytype_opt", &raytype_opt, "Specify ray type mask for optimization",
                "--iters %d", &iters, "Number of iterations",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--entry %L", &entrylayers, "Add layer to the list of entry points",
                "--entryoutput %L", &entryoutputs, "Add output symbol to the list of entry points",
                "--center", &pixelcenters, "Shade at output pixel 'centers' rather than corners",
                "--debugnan", &debugnan, "Turn on 'debug_nan' mode",
                "--debuguninit", &debug_uninit, "Turn on 'debug_uninit' mode",
                "--groupoutputs", &use_group_outputs, "Specify group outputs, not global outputs",
                "--oslquery", &do_oslquery, "Test OSLQuery at runtime",
                "--inbuffer", &inbuffer, "Compile osl source from and to buffer",
                "--shadeimage", &use_shade_image, "Use shade_image utility",
                "--noshadeimage %!", &use_shade_image, "Don't use shade_image utility",
                "--expr %@ %s", &specify_expr, NULL, "Specify an OSL expression to evaluate",
                "--offsetst %f %f", &soffset, &toffset, "Offset s & t texture coordinates (default: 0 0)",
                "--scalest %f %f", &sscale, &tscale, "Scale s & t texture lookups (default: 1, 1)",
                "--userdata_isconnected", &userdata_isconnected, "Consider lockgeom=0 to be isconnected()",
                "-v", &verbose, "Verbose output",
                NULL);
    if (ap.parse(argc, argv) < 0 || (shadernames.empty() && groupspec.empty())) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        std::cout << "testshade -- Test Open Shading Language\n"
                     OSL_COPYRIGHT_STRING "\n";
        ap.usage ();
        exit (EXIT_SUCCESS);
    }

    if (debug || verbose)
        errhandler.verbosity (ErrorHandler::VERBOSE);
    raytype_bit = shadingsys->raytype_bit (ustring (raytype));
}



// Here we set up transformations.  These are just examples, set up so
// that our unit tests can transform among spaces in ways that we will
// recognize as correct.  The "shader" and "object" spaces are required
// by OSL and the ShaderGlobals will need to have references to them.
// For good measure, we also set up a "myspace" space, registering it
// with the RendererServices.
// 
static void
setup_transformations (SimpleRenderer &rend, OSL::Matrix44 &Mshad,
                       OSL::Matrix44 &Mobj)
{
    Matrix44 M (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    rend.camera_params (M, ustring("perspective"), 90.0f,
                        0.1f, 1000.0f, xres, yres);

    // Make a "shader" space that is translated one unit in x and rotated
    // 45deg about the z axis.
    Mshad.makeIdentity ();
    Mshad.translate (OSL::Vec3 (1.0, 0.0, 0.0));
    Mshad.rotate (OSL::Vec3 (0.0, 0.0, M_PI_4));
    // std::cout << "shader-to-common matrix: " << Mshad << "\n";

    // Make an object space that is translated one unit in y and rotated
    // 90deg about the z axis.
    Mobj.makeIdentity ();
    Mobj.translate (OSL::Vec3 (0.0, 1.0, 0.0));
    Mobj.rotate (OSL::Vec3 (0.0, 0.0, M_PI_2));
    // std::cout << "object-to-common matrix: " << Mobj << "\n";

    OSL::Matrix44 Mmyspace;
    Mmyspace.scale (OSL::Vec3 (1.0, 2.0, 1.0));
    // std::cout << "myspace-to-common matrix: " << Mmyspace << "\n";
    rend.name_transform ("myspace", Mmyspace);
}



// Set up the ShaderGlobals fields for pixel (x,y).
static void
setup_shaderglobals (ShaderGlobals &sg, ShadingSystem *shadingsys,
                     int x, int y)
{
    // Just zero the whole thing out to start
    memset(&sg, 0, sizeof(ShaderGlobals));

    // In our SimpleRenderer, the "renderstate" itself just a pointer to
    // the ShaderGlobals.
    sg.renderstate = &sg;

    // Set "shader" space to be Mshad.  In a real renderer, this may be
    // different for each shader group.
    sg.shader2common = OSL::TransformationPtr (&Mshad);

    // Set "object" space to be Mobj.  In a real renderer, this may be
    // different for each object.
    sg.object2common = OSL::TransformationPtr (&Mobj);

    // Just make it look like all shades are the result of 'raytype' rays.
    sg.raytype = raytype_bit;

    // Set up u,v to vary across the "patch", and also their derivatives.
    // Note that since u & x, and v & y are aligned, we only need to set
    // values for dudx and dvdy, we can use the memset above to have set
    // dvdx and dudy to 0.
    if (pixelcenters) {
        // Our patch is like an "image" with shading samples at the
        // centers of each pixel.
        sg.u = sscale * (float)(x+0.5f) / xres + soffset;
        sg.v = tscale * (float)(y+0.5f) / yres + toffset;
        sg.dudx = sscale / xres;
        sg.dvdy = tscale / yres;
    } else {
        // Our patch is like a Reyes grid of points, with the border
        // samples being exactly on u,v == 0 or 1.
        sg.u = sscale * ((xres == 1) ? 0.5f : (float) x / (xres - 1)) + soffset;
        sg.v = tscale * ((yres == 1) ? 0.5f : (float) y / (yres - 1)) + toffset;
        sg.dudx = sscale / std::max (1, xres-1);
        sg.dvdy = tscale / std::max (1, yres-1);
    }

    // Assume that position P is simply (u,v,1), that makes the patch lie
    // on [0,1] at z=1.
    sg.P = Vec3 (sg.u, sg.v, 1.0f);
    // Derivatives with respect to x,y
    sg.dPdx = Vec3 (sg.dudx, sg.dudy, 0.0f);
    sg.dPdy = Vec3 (sg.dvdx, sg.dvdy, 0.0f);
    sg.dPdz = Vec3 (0.0f, 0.0f, 0.0f);  // just use 0 for volume tangent
    // Tangents of P with respect to surface u,v
    sg.dPdu = Vec3 (1.0f, 0.0f, 0.0f);
    sg.dPdv = Vec3 (0.0f, 1.0f, 0.0f);
    // That also implies that our normal points to (0,0,1)
    sg.N    = Vec3 (0, 0, 1);
    sg.Ng   = Vec3 (0, 0, 1);

    // Set the surface area of the patch to 1 (which it is).  This is
    // only used for light shaders that call the surfacearea() function.
    sg.surfacearea = 1;
}


// Set up the ShaderGlobals fields for pixel (x,y).
static void
setup_uniform_shaderglobals (ShaderGlobalsBatch &sgb, ShadingSystem *shadingsys)
{
	auto &usg = sgb.uniform();
	
    // Just zero the whole thing out to start
    memset(&usg, 0, sizeof(UniformShaderGlobals));

    // In our SimpleRenderer, the "renderstate" itself just a pointer to
    // the ShaderGlobals.
    usg.renderstate = &sgb;

    // Just make it look like all shades are the result of 'raytype' rays.
    usg.raytype = raytype_bit;
    
    
    // For this problem we will treat several varying members of 
    // the ShaderGlobalsBatch as uniform values.  We can assign to the 
    // proxy returned by helper method .uniform() to populate all varying
    // entries with a uniform value;
    auto vsp = sgb.varying();
    
    // Set "shader" space to be Mshad.  In a real renderer, this may be
    // different for each shader group.
    vsp.shader2common().uniform() = OSL::TransformationPtr (&Mshad);

    // Set "object" space to be Mobj.  In a real renderer, this may be
    // different for each object.
	vsp.object2common().uniform() = OSL::TransformationPtr (&Mobj);

    // Set up u,v to vary across the "patch", and also their derivatives.
    // Note that since u & x, and v & y are aligned, we only need to set
    // values for dudx and dvdy, we can use the memset above to have set
    // dvdx and dudy to 0.
    if (pixelcenters) {
        // Our patch is like an "image" with shading samples at the
        // centers of each pixel.
    	vsp.dudx().uniform() = sscale / xres;
    	vsp.dvdy().uniform() = tscale / yres;
    } else {
        // Our patch is like a Reyes grid of points, with the border
        // samples being exactly on u,v == 0 or 1.
    	vsp.dudx().uniform() = sscale / std::max (1, xres-1);
    	vsp.dvdy().uniform() = tscale / std::max (1, yres-1);
    }

    // Assume that position P is simply (u,v,1), that makes the patch lie
    // on [0,1] at z=1.
    // Derivatives with respect to x,y
    Vec3 dPdx = Vec3 (vsp.dudx(), vsp.dudy(), 0.0f);
    //vsp.dPdx().uniform() = Vec3 (vsp.dudx(), vsp.dudy(), 0.0f);
    vsp.dPdx().uniform() = dPdx; 
    vsp.dPdy().uniform() = Vec3 (vsp.dvdx(), vsp.dvdy(), 0.0f);
    vsp.dPdz().uniform() = Vec3 (0.0f, 0.0f, 0.0f);  // just use 0 for volume tangent
    // Tangents of P with respect to surface u,v
    vsp.dPdu().uniform() = Vec3 (1.0f, 0.0f, 0.0f);
    vsp.dPdv().uniform() = Vec3 (0.0f, 1.0f, 0.0f);
    // That also implies that our normal points to (0,0,1)
    vsp.N().uniform()    = Vec3 (0, 0, 1);
    vsp.Ng().uniform()   = Vec3 (0, 0, 1);

    // Set the surface area of the patch to 1 (which it is).  This is
    // only used for light shaders that call the surfacearea() function.
    vsp.surfacearea().uniform() = 1;    
}

static inline void
setup_varying_shaderglobals (ShaderGlobalsBatch & sgb, ShadingSystem *shadingsys,
                     int x, int y)
{
	auto vsp = sgb.varying();
	
    // Set up u,v to vary across the "patch", and also their derivatives.
    // Note that since u & x, and v & y are aligned, we only need to set
    // values for dudx and dvdy, we can use the memset above to have set
    // dvdx and dudy to 0.
	float u;
	float v;
    if (pixelcenters) {
        // Our patch is like an "image" with shading samples at the
        // centers of each pixel.
    	u = sscale * (float)(x+0.5f) / xres + soffset;
    	v = tscale * (float)(y+0.5f) / yres + toffset;
    } else {
        // Our patch is like a Reyes grid of points, with the border
        // samples being exactly on u,v == 0 or 1.
    	u = sscale * ((xres == 1) ? 0.5f : (float) x / (xres - 1)) + soffset;
    	v = tscale * ((yres == 1) ? 0.5f : (float) y / (yres - 1)) + toffset;
    }
 
    vsp.u() = u;
    vsp.v() = v;
    // Assume that position P is simply (u,v,1), that makes the patch lie
    // on [0,1] at z=1.
    vsp.P() = Vec3 (u, v, 1.0f);
    
    sgb.commitVarying();
    
}
 

static void
setup_output_images (ShadingSystem *shadingsys,
                     ShaderGroupRef &shadergroup)
{
    // Tell the shading system which outputs we want
    if (outputvars.size()) {
        std::vector<const char *> aovnames (outputvars.size());
        for (size_t i = 0; i < outputvars.size(); ++i) {
            ustring varname (outputvars[i]);
            aovnames[i] = varname.c_str();
            size_t dot = varname.find('.');
            if (dot != ustring::npos) {
                // If the name contains a dot, it's intended to be layer.symbol
                varname = ustring (varname, dot+1);
            }
        }
        shadingsys->attribute (use_group_outputs ? shadergroup.get() : NULL,
                               "renderer_outputs",
                               TypeDesc(TypeDesc::STRING,(int)aovnames.size()),
                               &aovnames[0]);
        if (use_group_outputs)
            std::cout << "Marking group outputs, not global renderer outputs.\n";
    }

    if (entrylayers.size()) {
        std::vector<const char *> layers;
        std::cout << "Entry layers:";
        for (size_t i = 0; i < entrylayers.size(); ++i) {
            ustring layername (entrylayers[i]);  // convert to ustring
            int layid = shadingsys->find_layer (*shadergroup, layername);
            layers.push_back (layername.c_str());
            entrylayer_index.push_back (layid);
            std::cout << ' ' << entrylayers[i] << "(" << layid << ")";
        }
        std::cout << "\n";
        shadingsys->attribute (shadergroup.get(), "entry_layers",
                               TypeDesc(TypeDesc::STRING,(int)entrylayers.size()),
                               &layers[0]);
    }

    if (extraoptions.size())
        shadingsys->attribute ("options", extraoptions);

    ShadingContext *ctx = shadingsys->get_context ();
    // Because we can only call find_symbol or get_symbol on something that
    // has been set up to shade (or executed), we call execute() but tell it
    // not to actually run the shader.
#if OSL_USE_WIDE_LLVM_BACKEND
    ShaderGlobalsBatch sgBatch;
    setup_uniform_shaderglobals (sgBatch, shadingsys);
	setup_varying_shaderglobals (sgBatch, shadingsys, 0, 0);
            
#else
    ShaderGlobals sg;
    setup_shaderglobals (sg, shadingsys, 0, 0);
#endif
    if (raytype_opt)
        shadingsys->optimize_group (shadergroup.get(), raytype_bit, ~raytype_bit);

    
#if OSL_USE_WIDE_LLVM_BACKEND
    shadingsys->execute_batch (ctx, *shadergroup, sgBatch, false);
#else
    shadingsys->execute (ctx, *shadergroup, sg, false);
#endif

    if (entryoutputs.size()) {
        std::cout << "Entry outputs:";
        for (size_t i = 0; i < entryoutputs.size(); ++i) {
            ustring name (entryoutputs[i]);  // convert to ustring
            const ShaderSymbol *sym = shadingsys->find_symbol (*shadergroup, name);
            if (!sym) {
                std::cout << "\nEntry output " << entryoutputs[i] << " not found. Abording.\n";
                exit (EXIT_FAILURE);
            }
            entrylayer_symbols.push_back (sym);
            std::cout << ' ' << entryoutputs[i];
        }
        std::cout << "\n";
    }

    // For each output file specified on the command line...
    for (size_t i = 0;  i < outputfiles.size();  ++i) {
        // Make a ustring version of the output name, for fast manipulation
        outputvarnames.push_back (ustring(outputvars[i]));
        // Start with a NULL ImageBuf pointer
        outputimgs.push_back (NULL);

        // Ask for a pointer to the symbol's data, as computed by this
        // shader.
        TypeDesc t;
        const void *data = shadingsys->get_symbol (*ctx, outputvarnames[i], t);
        if (!data) {
            std::cout << "Output " << outputvars[i] 
                      << " not found, skipping.\n";
            continue;  // Skip if symbol isn't found
        }
        std::cout << "Output " << outputvars[i] << " to "
                  << outputfiles[i] << "\n";

        // And the "base" type, i.e. the type of each element or channel
        TypeDesc tbase = TypeDesc ((TypeDesc::BASETYPE)t.basetype);

        // But which type are we going to write?  Use the true data type
        // from OSL, unless the command line options indicated that
        // something else was desired.
        TypeDesc outtypebase = tbase;
        if (dataformatname == "uint8")
            outtypebase = TypeDesc::UINT8;
        else if (dataformatname == "half")
            outtypebase = TypeDesc::HALF;
        else if (dataformatname == "float")
            outtypebase = TypeDesc::FLOAT;

        // Number of channels to write to the image is the number of (array)
        // elements times the number of channels (e.g. 1 for scalar, 3 for
        // vector, etc.)
        int nchans = t.numelements() * t.aggregate;

        // Make an ImageBuf of the right type and size to hold this
        // symbol's output, and initially clear it to all black pixels.
        OIIO::ImageSpec spec (xres, yres, nchans, TypeDesc::FLOAT);
        outputimgs[i] = new OIIO::ImageBuf(outputfiles[i], spec);
        outputimgs[i]->set_write_format (outtypebase);
        OIIO::ImageBufAlgo::zero (*outputimgs[i]);
    }

    if (outputimgs.empty()) {
        OIIO::ImageSpec spec (xres, yres, 3, TypeDesc::FLOAT);
        outputimgs.push_back(new OIIO::ImageBuf(spec));
    }

    shadingsys->release_context (ctx);  // don't need this anymore for now
}



// For pixel (x,y) that was just shaded by the given shading context,
// save each of the requested outputs to the corresponding output
// ImageBuf.
//
// In a real renderer, this is illustrative of how you would pull shader
// outputs into "AOV's" (arbitrary output variables, or additional
// renderer outputs).  You would, of course, also grab the closure Ci
// and integrate the lights using that BSDF to determine the radiance
// in the direction of the camera for that pixel.
static void
save_outputs (ShadingSystem *shadingsys, ShadingContext *ctx, int x, int y)
{
    // For each output requested on the command line...
    for (size_t i = 0;  i < outputfiles.size();  ++i) {
        // Skip if we couldn't open the image or didn't match a known output
        if (! outputimgs[i])
            continue;

        // Ask for a pointer to the symbol's data, as computed by this
        // shader.
        TypeDesc t;
        const void *data = shadingsys->get_symbol (*ctx, outputvarnames[i], t);
        if (!data)
            continue;  // Skip if symbol isn't found

        if (t.basetype == TypeDesc::FLOAT) {
            // If the variable we are outputting is float-based, set it
            // directly in the output buffer.
			//printf("xy=(%d,%d) = (%f,%f,%f)\n", x, y, ((const float *)data)[0], ((const float *)data)[1], ((const float *)data)[2]);
            outputimgs[i]->setpixel (x, y, (const float *)data);
        } else if (t.basetype == TypeDesc::INT) {
            // We are outputting an integer variable, so we need to
            // convert it to floating point.
            int nchans = outputimgs[i]->nchannels();
            float *pixel = (float *) alloca (nchans * sizeof(float));
            OIIO::convert_types (TypeDesc::BASETYPE(t.basetype), data,
                                 TypeDesc::FLOAT, pixel, nchans);
            outputimgs[i]->setpixel (x, y, &pixel[0]);
        }
        // N.B. Drop any outputs that aren't float- or int-based
    }
}

// For pixel (x,y) that was just shaded by the given shading context,
// save each of the requested outputs to the corresponding output
// ImageBuf.
//
// In a real renderer, this is illustrative of how you would pull shader
// outputs into "AOV's" (arbitrary output variables, or additional
// renderer outputs).  You would, of course, also grab the closure Ci
// and integrate the lights using that BSDF to determine the radiance
// in the direction of the camera for that pixel.
static void
batched_save_outputs (ShadingSystem *shadingsys, ShadingContext *ctx, ShaderGroup *shadergroup, int x, int y, int batchIndex)
{
    // For each output requested on the command line...
    for (size_t i = 0;  i < outputfiles.size();  ++i) {
        // Skip if we couldn't open the image or didn't match a known output
        if (! outputimgs[i])
            continue;

		const ShaderSymbol* out_symbol = shadingsys->find_symbol (*shadergroup, outputvarnames[i]);
        if (!out_symbol)
            continue;  // Skip if symbol isn't found
        
        TypeDesc t = shadingsys->symbol_typedesc(out_symbol);

        // Ask for a batch accessor to the symbol's data, as computed by this
        // shader.
        
        
        if (t.basetype == TypeDesc::FLOAT) {
        	if (t.aggregate == TypeDesc::VEC3) {        	
				// If the variable we are outputting is float-based, set it
				// directly in the output buffer.
				auto batchResults = shadingsys->symbol_batch_accessor<Vec3>(*ctx, out_symbol);
				Vec3 data = batchResults[batchIndex];
				//printf("xy=(%d,%d) = (%f,%f,%f)\n", x, y, data[0], data[1], data[2]);
				outputimgs[i]->setpixel (x, y, reinterpret_cast<const float *>(&data));
        	}
        } else if (t.basetype == TypeDesc::INT) {
            auto batchResults = shadingsys->symbol_batch_accessor<int>(*ctx, out_symbol);
        	int data = batchResults[batchIndex];
            // We are outputting an integer variable, so we need to
            // convert it to floating point.
            int nchans = outputimgs[i]->nchannels();
            float *pixel = (float *) alloca (nchans * sizeof(float));
            OIIO::convert_types (TypeDesc::BASETYPE(t.basetype), &data,
                                 TypeDesc::FLOAT, pixel, nchans);
            outputimgs[i]->setpixel (x, y, &pixel[0]);
        }
        // N.B. Drop any outputs that aren't float- or int-based
    }
}


static void
test_group_attributes (ShaderGroup *group)
{
    int nt = 0;
    if (shadingsys->getattribute (group, "num_textures_needed", nt)) {
        std::cout << "Need " << nt << " textures:\n";
        ustring *tex = NULL;
        shadingsys->getattribute (group, "textures_needed",
                                  TypeDesc::PTR, &tex);
        for (int i = 0; i < nt; ++i)
            std::cout << "    " << tex[i] << "\n";
        int unk = 0;
        shadingsys->getattribute (group, "unknown_textures_needed", unk);
        if (unk)
            std::cout << "    and unknown textures\n";
    }
    int nclosures = 0;
    if (shadingsys->getattribute (group, "num_closures_needed", nclosures)) {
        std::cout << "Need " << nclosures << " closures:\n";
        ustring *closures = NULL;
        shadingsys->getattribute (group, "closures_needed",
                                  TypeDesc::PTR, &closures);
        for (int i = 0; i < nclosures; ++i)
            std::cout << "    " << closures[i] << "\n";
        int unk = 0;
        shadingsys->getattribute (group, "unknown_closures_needed", unk);
        if (unk)
            std::cout << "    and unknown closures\n";
    }
    int nglobals = 0;
    if (shadingsys->getattribute (group, "num_globals_needed", nglobals)) {
        std::cout << "Need " << nglobals << " globals:\n";
        ustring *globals = NULL;
        shadingsys->getattribute (group, "globals_needed",
                                  TypeDesc::PTR, &globals);
        for (int i = 0; i < nglobals; ++i)
            std::cout << "    " << globals[i] << "\n";
    }
    int nuser = 0;
    if (shadingsys->getattribute (group, "num_userdata", nuser) && nuser) {
        std::cout << "Need " << nuser << " user data items:\n";
        ustring *userdata_names = NULL;
        TypeDesc *userdata_types = NULL;
        int *userdata_offsets = NULL;
        bool *userdata_derivs = NULL;
        shadingsys->getattribute (group, "userdata_names",
                                  TypeDesc::PTR, &userdata_names);
        shadingsys->getattribute (group, "userdata_types",
                                  TypeDesc::PTR, &userdata_types);
        shadingsys->getattribute (group, "userdata_offsets",
                                  TypeDesc::PTR, &userdata_offsets);
        shadingsys->getattribute (group, "userdata_derivs",
                                  TypeDesc::PTR, &userdata_derivs);
        DASSERT (userdata_names && userdata_types && userdata_offsets);
        for (int i = 0; i < nuser; ++i)
            std::cout << "    " << userdata_names[i] << ' '
                      << userdata_types[i] << "  offset="
                      << userdata_offsets[i] << " deriv="
                      << userdata_derivs[i] << "\n";
    }
    int nattr = 0;
    if (shadingsys->getattribute (group, "num_attributes_needed", nattr) && nattr) {
        std::cout << "Need " << nattr << " attributes:\n";
        ustring *names = NULL;
        ustring *scopes = NULL;
        shadingsys->getattribute (group, "attributes_needed",
                                  TypeDesc::PTR, &names);
        shadingsys->getattribute (group, "attribute_scopes",
                                  TypeDesc::PTR, &scopes);
        DASSERT (names && scopes);
        for (int i = 0; i < nattr; ++i)
            std::cout << "    " << names[i] << ' '
                      << scopes[i] << "\n";

        int unk = 0;
        shadingsys->getattribute (group, "unknown_attributes_needed", unk);
        if (unk)
            std::cout << "    and unknown attributes\n";
    }
    int raytype_queries = 0;
    shadingsys->getattribute (group, "raytype_queries", raytype_queries);
    std::cout << "raytype() query mask: " << raytype_queries << "\n";
}



void
shade_region (ShaderGroup *shadergroup, OIIO::ROI roi, bool save)
{
	std::cout << "shade_region(roi.x[" << roi.xbegin << "," << roi.xend << ") roi.y[" << roi.ybegin << "," << roi.yend << ")" << std::endl;
    // Optional: high-performance apps may request this thread-specific
    // pointer in order to save a bit of time on each shade.  Just like
    // the name implies, a multithreaded renderer would need to do this
    // separately for each thread, and be careful to always use the same
    // thread_info each time for that thread.
    //
    // There's nothing wrong with a simpler app just passing NULL for
    // the thread_info; in such a case, the ShadingSystem will do the
    // necessary calls to find the thread-specific pointer itself, but
    // this will degrade performance just a bit.
    OSL::PerThreadInfo *thread_info = shadingsys->create_thread_info();

    // Request a shading context so that we can execute the shader.
    // We could get_context/release_constext for each shading point,
    // but to save overhead, it's more efficient to reuse a context
    // within a thread.
    ShadingContext *ctx = shadingsys->get_context (thread_info);

    // Set up shader globals and a little test grid of points to shade.
    ShaderGlobals shaderglobals;

    // Loop over all pixels in the image (in x and y)...
    for (int y = roi.ybegin;  y < roi.yend;  ++y) {
        for (int x = roi.xbegin;  x < roi.xend;  ++x) {
            // In a real renderer, this is where you would figure
            // out what object point is visible in this pixel (or
            // this sample, for antialiasing).  Once determined,
            // you'd set up a ShaderGlobals that contained the vital
            // information about that point, such as its location,
            // the normal there, the u and v coordinates on the
            // surface, the transformation of that object, and so
            // on.  
            //
            // This test app is not a real renderer, so we just
            // set it up rigged to look like we're rendering a single
            // quadrilateral that exactly fills the viewport, and that
            // setup is done in the following function call:
            setup_shaderglobals (shaderglobals, shadingsys, x, y);

            // Actually run the shader for this point
            if (entrylayer_index.empty()) {
                // Sole entry point for whole group, default behavior
                shadingsys->execute (ctx, *shadergroup, shaderglobals);
            } else {
                // Explicit list of entries to call in order
                shadingsys->execute_init (*ctx, *shadergroup, shaderglobals);
                if (entrylayer_symbols.size()) {
                    for (size_t i = 0, e = entrylayer_symbols.size(); i < e; ++i)
                        shadingsys->execute_layer (*ctx, shaderglobals, entrylayer_symbols[i]);
                } else {
                    for (size_t i = 0, e = entrylayer_index.size(); i < e; ++i)
                        shadingsys->execute_layer (*ctx, shaderglobals, entrylayer_index[i]);
                }
                shadingsys->execute_cleanup (*ctx);
            }

            // Save all the designated outputs.  But only do so if we
            // are on the last iteration requested, so that if we are
            // doing a bunch of iterations for time trials, we only
            // including the output pixel copying once in the timing.
            if (save)
                save_outputs (shadingsys, ctx, x, y);
        }
    }

    // We're done shading with this context.
    shadingsys->release_context (ctx);

    // Now that we're done rendering, release the thread-specific
    // pointer we saved.  A simple app could skip this; but if the app
    // asks for it (as we have in this example), then it should also
    // destroy it when done with it.
    shadingsys->destroy_thread_info(thread_info);
}


void
batched_shade_region (ShaderGroup *shadergroup, OIIO::ROI roi, bool save)
{
    // Optional: high-performance apps may request this thread-specific
    // pointer in order to save a bit of time on each shade.  Just like
    // the name implies, a multithreaded renderer would need to do this
    // separately for each thread, and be careful to always use the same
    // thread_info each time for that thread.
    //
    // There's nothing wrong with a simpler app just passing NULL for
    // the thread_info; in such a case, the ShadingSystem will do the
    // necessary calls to find the thread-specific pointer itself, but
    // this will degrade performance just a bit.
    OSL::PerThreadInfo *thread_info = shadingsys->create_thread_info();

    // Request a shading context so that we can execute the shader.
    // We could get_context/release_constext for each shading point,
    // but to save overhead, it's more efficient to reuse a context
    // within a thread.
    ShadingContext *ctx = shadingsys->get_context (thread_info);

    // Set up shader globals and a little test grid of points to shade.
    ShaderGlobalsBatch sgBatch;
    setup_uniform_shaderglobals (sgBatch, shadingsys);
    
//    std::cout << "shading roi y(" << roi.ybegin << ", " << roi.yend << ")";
//    std::cout << " x(" << roi.xbegin << ", " << roi.xend << ")" << std::endl;
    
    // Seperate loop indices to pull results out of batch
    // These result indices follow along after a batch is 
    // executed.
    int ry = roi.ybegin;
    int rx = roi.xbegin;
    
    // Loop over all pixels in the image (in x and y)...
    for (int y = roi.ybegin;  y < roi.yend;  ++y) {
    	bool isFinalY = (y == (roi.yend-1));
        for (int x = roi.xbegin;  x < roi.xend;  ++x) {
        	bool isFinalX = (x == (roi.xend-1));
            // In a real renderer, this is where you would figure
            // out what object point is visible in this pixel (or
            // this sample, for antialiasing).  Once determined,
            // you'd set up a ShaderGlobalsBatch that contained the vital
            // information about that point, such as its location,
            // the normal there, the u and v coordinates on the
            // surface, the transformation of that object, and so
            // on.  
            //
            // This test app is not a real renderer, so we just
            // set it up rigged to look like we're rendering a single
            // quadrilateral that exactly fills the viewport, and that
            // setup is done in the following function call:
        	setup_varying_shaderglobals (sgBatch, shadingsys, x, y);

            //std::cout << "shading x=" << x << " y=" << y << std::endl;
            if(sgBatch.isFull() || (isFinalX && isFinalY))
            {
	            //std::cout << "shading batch with size=" << sgBatch.size() << std::endl;
				
				// Actually run the shader for this point
				if (entrylayer_index.empty()) {
					// Sole entry point for whole group, default behavior
					shadingsys->execute_batch (ctx, *shadergroup, sgBatch);
				} else {
					ASSERT(false && "untested");
					// Explicit list of entries to call in order
					shadingsys->execute_batch_init (*ctx, *shadergroup, sgBatch);
					if (entrylayer_symbols.size()) {
						for (size_t i = 0, e = entrylayer_symbols.size(); i < e; ++i)
							shadingsys->execute_batch_layer (*ctx, sgBatch, entrylayer_symbols[i]);
					} else {
						for (size_t i = 0, e = entrylayer_index.size(); i < e; ++i)
							shadingsys->execute_batch_layer (*ctx, sgBatch, entrylayer_index[i]);
					}
					shadingsys->execute_cleanup (*ctx);
				}
				
				
				if (save)
				{
					int bs = sgBatch.size();
					for(int bi=0; bi < bs; ++bi)
					{
						// Save all the designated outputs.  But only do so if we
						// are on the last iteration requested, so that if we are
						// doing a bunch of iterations for time trials, we only
						// including the output pixel copying once in the timing.
						
						batched_save_outputs (shadingsys, ctx, shadergroup, rx, ry, bi);
						if(++rx >= roi.xend) {
							rx = roi.xbegin;
							++ry;
						}
					}
				}
				sgBatch.clear();
            }
        }
    }

    if(sgBatch.isEmpty() == false)
    {
    	std::cerr << "Unsubmitted batch, probably forgot to handle the final batch which may not be full" << std::endl;
    }
    
    // We're done shading with this context.
    shadingsys->release_context (ctx);
 
    // Now that we're done rendering, release the thread-specific
    // pointer we saved.  A simple app could skip this; but if the app
    // asks for it (as we have in this example), then it should also
    // destroy it when done with it.
    shadingsys->destroy_thread_info(thread_info);
}


extern "C" OSL_DLL_EXPORT int
test_shade (int argc, const char *argv[])
{
    
    OIIO::Timer timer;

    // Create a new shading system.  We pass it the RendererServices
    // object that services callbacks from the shading system, NULL for
    // the TextureSystem (that just makes 'create' make its own TS), and
    // an error handler.
    shadingsys = new ShadingSystem (&rend, NULL, &errhandler);

    // Register the layout of all closures known to this renderer
    // Any closure used by the shader which is not registered, or
    // registered with a different number of arguments will lead
    // to a runtime error.
    register_closures(shadingsys);

    // Remember that each shader parameter may optionally have a
    // metadata hint [[int lockgeom=...]], where 0 indicates that the
    // parameter may be overridden by the geometry itself, for example
    // with data interpolated from the mesh vertices, and a value of 1
    // means that it is "locked" with respect to the geometry (i.e. it
    // will not be overridden with interpolated or
    // per-geometric-primitive data).
    // 
    // In order to most fully optimize shader, we typically want any
    // shader parameter not explicitly specified to default to being
    // locked (i.e. no per-geometry override):
    shadingsys->attribute("lockgeom", 1);

    // Now we declare our shader.
    // 
    // Each material in the scene is comprised of a "shader group."
    // Each group is comprised of one or more "layers" (a.k.a. shader
    // instances) with possible connections from outputs of
    // upstream/early layers into the inputs of downstream/later layers.
    // A shader instance is the combination of a reference to a shader
    // master and its parameter values that may override the defaults in
    // the shader source and may be particular to this instance (versus
    // all the other instances of the same shader).
    // 
    // A shader group declaration typically looks like this:
    //
    //   ShaderGroupRef shadergroup = ss->ShaderGroupBegin ();
    //   ss->Parameter ("paramname", TypeDesc paramtype, void *value);
    //      ... and so on for all the other parameters of...
    //   ss->Shader ("shadertype", "shadername", "layername");
    //      The Shader() call creates a new instance, which gets
    //      all the pending Parameter() values made right before it.
    //   ... and other shader instances in this group, interspersed with...
    //   ss->ConnectShaders ("layer1", "param1", "layer2", "param2");
    //   ... and other connections ...
    //   ss->ShaderGroupEnd ();
    // 
    // It looks so simple, and it really is, except that the way this
    // testshade program works is that all the Parameter() and Shader()
    // calls are done inside getargs(), as it walks through the command
    // line arguments, whereas the connections accumulate and have
    // to be processed at the end.  Bear with us.
    
    // Start the shader group and grab a reference to it.
    shadergroup = shadingsys->ShaderGroupBegin (groupname);

    // Get the command line arguments.  That will set up all the shader
    // instances and their parameters for the group.
    getargs (argc, argv);

    if (! shadergroup) {
        std::cerr << "ERROR: Invalid shader group. Exiting testshade.\n";
        return EXIT_FAILURE;
    }

    shadingsys->attribute (shadergroup.get(), "groupname", groupname);

    // Now set up the connections
    for (size_t i = 0;  i < connections.size();  i += 4) {
        if (i+3 < connections.size()) {
            std::cout << "Connect " 
                      << connections[i] << "." << connections[i+1]
                      << " to " << connections[i+2] << "." << connections[i+3]
                      << "\n";
            shadingsys->ConnectShaders (connections[i].c_str(),
                                        connections[i+1].c_str(),
                                        connections[i+2].c_str(),
                                        connections[i+3].c_str());
        }
    }

    // End the group
    shadingsys->ShaderGroupEnd ();

    if (verbose || do_oslquery) {
        std::string pickle;
        shadingsys->getattribute (shadergroup.get(), "pickle", pickle);
        std::cout << "Shader group:\n---\n" << pickle << "\n---\n";
        std::cout << "\n";
        ustring groupname;
        shadingsys->getattribute (shadergroup.get(), "groupname", groupname);
        std::cout << "Shader group \"" << groupname << "\" layers are:\n";
        int num_layers = 0;
        shadingsys->getattribute (shadergroup.get(), "num_layers", num_layers);
        if (num_layers > 0) {
            std::vector<const char *> layers (size_t(num_layers), NULL);
            shadingsys->getattribute (shadergroup.get(), "layer_names",
                                      TypeDesc(TypeDesc::STRING, num_layers),
                                      &layers[0]);
            for (int i = 0; i < num_layers; ++i) {
                std::cout << "    " << (layers[i] ? layers[i] : "<unnamed>") << "\n";
                if (do_oslquery) {
                    OSLQuery q;
                    q.init (shadergroup.get(), i);
                    for (size_t p = 0;  p < q.nparams(); ++p) {
                        const OSLQuery::Parameter *param = q.getparam(p);
                        std::cout << "\t" << (param->isoutput ? "output "  : "")
                                  << param->type << ' ' << param->name << "\n";
                    }
                }
            }
        }
        std::cout << "\n";
    }
    if (archivegroup.size())
        shadingsys->archive_shadergroup (shadergroup.get(), archivegroup);

    if (outputfiles.size() != 0)
        std::cout << "\n";

    // Set up the named transformations, including shader and object.
    // For this test application, we just do this statically; in a real
    // renderer, the global named space (like "myspace") would probably
    // be static, but shader and object spaces may be different for each
    // object.
    setup_transformations (rend, Mshad, Mobj);

    if (num_threads < 1) {
    	// TODO:  number of logical cores isn't a good choice
    	// as taskset or numactl or mpi may have launched this process with
    	// fewer threads.  This can lead to oversubscription.
        num_threads = OIIO::Sysutil::hardware_concurrency();
    }
    std::cout << "num_threads = " << num_threads << std::endl;

    // We need to set the global attribute so any helper functions
    // respect our thread count, especially if we wanted only 1
    // thread, we want to avoid spinning up a thread pool or
    // OS overhead of destroying threads (like clearing virtual 
    // memory pages they occupied)
    OIIO::attribute("threads",num_threads);
    
    // Set up the image outputs requested on the command line
    setup_output_images (shadingsys, shadergroup);
    

    if (debug)
        test_group_attributes (shadergroup.get());


    double setuptime = timer.lap ();

    // Allow a settable number of iterations to "render" the whole image,
    // which is useful for time trials of things that would be too quick
    // to accurately time for a single iteration
    for (int iter = 0;  iter < iters;  ++iter) {
        OIIO::ROI roi (0, xres, 0, yres);

        if (use_shade_image)
            OSL::shade_image (*shadingsys, *shadergroup, NULL,
                              *outputimgs[0], outputvarnames,
                              pixelcenters ? ShadePixelCenters : ShadePixelGrid,
                              roi, num_threads);
        else {
            bool save = (iter == (iters-1));   // save on last iteration
#if OSL_USE_WIDE_LLVM_BACKEND
            batched_shade_region (shadergroup.get(), roi, save);
#else
	#if 0
            shade_region (shadergroup.get(), roi, save);
	#else
            OIIO::ImageBufAlgo::parallel_image (
                    std::bind (shade_region, shadergroup.get(), std::placeholders::_1, save),
                    roi, num_threads);
	#endif
#endif
        }

        // If any reparam was requested, do it now
        if (reparams.size() && reparam_layer.size()) {
            for (size_t p = 0;  p < reparams.size();  ++p) {
                const ParamValue &pv (reparams[p]);
                shadingsys->ReParameter (*shadergroup, reparam_layer.c_str(),
                                         pv.name().c_str(), pv.type(),
                                         pv.data());
            }
        }
    }

    if (outputfiles.size() == 0)
        std::cout << "\n";

    // Write the output images to disk
    for (size_t i = 0;  i < outputimgs.size();  ++i) {
        if (outputimgs[i]) {
            outputimgs[i]->write (outputimgs[i]->name());
            delete outputimgs[i];
            outputimgs[i] = NULL;
        }
    }

    // Print some debugging info
    if (debug || runstats || profile) {
        double runtime = timer.lap();
        std::cout << "\n";
        std::cout << "Setup: " << OIIO::Strutil::timeintervalformat (setuptime,2) << "\n";
        std::cout << "Run  : " << OIIO::Strutil::timeintervalformat (runtime,2) << "\n";
        std::cout << "\n";
        std::cout << shadingsys->getstats (5) << "\n";
        OIIO::TextureSystem *texturesys = shadingsys->texturesys();
        if (texturesys)
            std::cout << texturesys->getstats (5) << "\n";
        std::cout << ustring::getstats() << "\n";
    }

    // We're done with the shading system now, destroy it
    shadergroup.reset ();  // Must release this before destroying shadingsys
    delete shadingsys;

    return EXIT_SUCCESS;
}
