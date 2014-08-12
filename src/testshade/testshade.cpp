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

#include <boost/foreach.hpp>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagebufalgo_util.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>

#include "OSL/oslexec.h"
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
static bool debug = false;
static bool debug2 = false;
static bool verbose = false;
static bool stats = false;
static bool O0 = false, O1 = false, O2 = false;
static bool pixelcenters = false;
static bool debugnan = false;
static bool debug_uninit = false;
static bool use_group_outputs = false;
static bool do_oslquery = false;
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
static std::string extraoptions;
static SimpleRenderer rend;  // RendererServices
static OSL::Matrix44 Mshad;  // "shader" space to "common" space matrix
static OSL::Matrix44 Mobj;   // "object" space to "common" space matrix
static ShaderGroupRef shadergroup;
static std::string archivegroup;


static void
inject_params ()
{
    for (size_t p = 0;  p < params.size();  ++p) {
        const ParamValue &pv (params[p]);
        shadingsys->Parameter (pv.name().c_str(), pv.type(), pv.data(),
                               pv.interp() == ParamValue::INTERP_CONSTANT);
    }
}



static int
add_shader (int argc, const char *argv[])
{
    shadingsys->attribute ("debug", debug2 ? 2 : (debug ? 1 : 0));
    shadingsys->attribute ("compile_report", debug|debug2);
    const char *opt_env = getenv ("TESTSHADE_OPT");  // overrides opt
    if (opt_env)
        shadingsys->attribute ("optimize", atoi(opt_env));
    else if (O0 || O1 || O2)
        shadingsys->attribute ("optimize", O2 ? 2 : (O1 ? 1 : 0));
    shadingsys->attribute ("lockgeom", 1);
    shadingsys->attribute ("debug_nan", debugnan);
    shadingsys->attribute ("debug_uninit", debug_uninit);

    for (int i = 0;  i < argc;  i++) {
        inject_params ();

        shadernames.push_back (argv[i]);
        shadingsys->Shader ("surface", argv[i],
                            layername.length() ? layername.c_str() : NULL);

        layername.clear ();
        params.clear ();
    }
    return 0;
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

    std::string paramname = argv[1];
    std::string stringval = argv[2];
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
    if (verbose)
        std::cout << "Processing group specification:\n---\n"
                  << argv[1] << "\n---\n";
    shadergroup = shadingsys->ShaderGroupBegin (groupname, "surface", argv[1]);
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
                "--stats", &stats, "Print run statistics",
                "-g %d %d", &xres, &yres, "Make an X x Y grid of shading points",
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
                "--iters %d", &iters, "Number of iterations",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--center", &pixelcenters, "Shade at output pixel 'centers' rather than corners",
                "--debugnan", &debugnan, "Turn on 'debug_nan' mode",
                "--debuguninit", &debug_uninit, "Turn on 'debug_uninit' mode",
                "--options %s", &extraoptions, "Set extra OSL options",
                "--groupoutputs", &use_group_outputs, "Specify group outputs, not global outputs",
                "--oslquery", &do_oslquery, "Test OSLQuery at runtime",
//                "-v", &verbose, "Verbose output",
                NULL);
    if (ap.parse(argc, argv) < 0 || (shadernames.empty() && groupspec.empty())) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        std::cout <<
            "testshade -- Test Open Shading Language\n"
            "(c) Copyright 2009-2010 Sony Pictures Imageworks Inc. All Rights Reserved.\n";
        ap.usage ();
        exit (EXIT_SUCCESS);
    }

    if (debug || verbose)
        errhandler.verbosity (ErrorHandler::VERBOSE);
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
    sg.raytype = shadingsys->raytype_bit (ustring(raytype));

    // Set up u,v to vary across the "patch", and also their derivatives.
    // Note that since u & x, and v & y are aligned, we only need to set
    // values for dudx and dvdy, we can use the memset above to have set
    // dvdx and dudy to 0.
    if (pixelcenters) {
        // Our patch is like an "image" with shading samples at the
        // centers of each pixel.
        sg.u = (float)(x+0.5f) / xres;
        sg.v = (float)(y+0.5f) / yres;
        sg.dudx = 1.0f / xres;
        sg.dvdy = 1.0f / yres;
    } else {
        // Our patch is like a Reyes grid of points, with the border
        // samples being exactly on u,v == 0 or 1.
        sg.u = (xres == 1) ? 0.5f : (float) x / (xres - 1);
        sg.v = (yres == 1) ? 0.5f : (float) y / (yres - 1);
        sg.dudx = 1.0f / std::max (1, xres-1);
        sg.dvdy = 1.0f / std::max (1, yres-1);
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



static void
setup_output_images (ShadingSystem *shadingsys,
                     ShaderGroupRef &shadergroup)
{
    // Tell the shading system which outputs we want
    if (outputvars.size()) {
        std::vector<const char *> aovnames (outputvars.size());
        for (size_t i = 0; i < outputvars.size(); ++i)
            aovnames[i] = outputvars[i].c_str();
        shadingsys->attribute (use_group_outputs ? shadergroup.get() : NULL,
                               "renderer_outputs",
                               TypeDesc(TypeDesc::STRING,(int)aovnames.size()),
                               &aovnames[0]);
        if (use_group_outputs)
            std::cout << "Marking group outputs, not global renderer outputs.\n";
    }

    if (extraoptions.size())
        shadingsys->attribute ("options", extraoptions);

    ShadingContext *ctx = shadingsys->get_context ();
    // Because we can only call get_symbol on something that has been
    // set up to shade (or executed), we call execute() but tell it not
    // to actually run the shader.
    ShaderGlobals sg;
    setup_shaderglobals (sg, shadingsys, 0, 0);
    shadingsys->execute (*ctx, *shadergroup, sg, false);

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
        OIIO::ImageSpec spec (xres, yres, nchans, outtypebase);
        outputimgs[i] = new OIIO::ImageBuf(outputfiles[i], spec);
        OIIO::ImageBufAlgo::zero (*outputimgs[i]);
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
}



static void
shade_region (ShaderGroup *shadergroup, OIIO::ROI roi, bool save)
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
            shadingsys->execute (*ctx, *shadergroup, shaderglobals);

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

    // Now that we're done rendering, release the thread=specific
    // pointer we saved.  A simple app could skip this; but if the app
    // asks for it (as we have in this example), then it should also
    // destroy it when done with it.
    shadingsys->destroy_thread_info(thread_info);
}



extern "C" int
test_shade (int argc, const char *argv[])
{
    OIIO::Timer timer;

    // Create a new shading system.  We pass it the RendererServices
    // object that services callbacks from the shading system, NULL for
    // the TextureSystem (that just makes 'create' make its own TS), and
    // an error handler.
    shadingsys = new ShadingSystem (&rend, NULL, &errhandler);
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
    shadergroup = shadingsys->ShaderGroupBegin ();

    // Get the command line arguments.  That will set up all the shader
    // instances and their parameters for the group.
    getargs (argc, argv);

    if (! shadergroup) {
        std::cerr << "ERROR: Invalid shader group. Exiting testshade.\n";
        return EXIT_FAILURE;
    }

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
            std::vector<const char *> layers (num_layers, NULL);
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

    // Set up the image outputs requested on the command line
    setup_output_images (shadingsys, shadergroup);

    if (debug)
        test_group_attributes (shadergroup.get());

    if (num_threads < 1)
        num_threads = boost::thread::hardware_concurrency();

    double setuptime = timer.lap ();

    // Allow a settable number of iterations to "render" the whole image,
    // which is useful for time trials of things that would be too quick
    // to accurately time for a single iteration
    for (int iter = 0;  iter < iters;  ++iter) {
        OIIO::ROI roi (0, xres, 0, yres);
        bool save = (iter == (iters-1));   // save on last iteration

#if 0
        shade_region (shadergroup.get(), roi, save);
#else
        OIIO::ImageBufAlgo::parallel_image (
            boost::bind (shade_region, shadergroup.get(), _1, save),
            roi, num_threads);
#endif

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
            outputimgs[i]->save();
            delete outputimgs[i];
            outputimgs[i] = NULL;
        }
    }

    // Print some debugging info
    if (debug || stats) {
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
