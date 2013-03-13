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
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>

#include "oslexec.h"
#include "simplerend.h"
using namespace OSL;



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
static int xres = 1, yres = 1;
static std::string layername;
static std::vector<std::string> connections;
static std::vector<std::string> iparams, fparams, vparams, sparams;
static float fparamdata[1000];   // bet that's big enough
static int fparamindex = 0;
static int iparamdata[1000];
static int iparamindex = 0;
static ustring sparamdata[1000];
static int sparamindex = 0;
static ErrorHandler errhandler;
static int iters = 1;
static std::string raytype = "camera";
static std::string extraoptions;
static SimpleRenderer rend;  // RendererServices
static OSL::Matrix44 Mshad;  // "shader" space to "common" space matrix
static OSL::Matrix44 Mobj;   // "object" space to "common" space matrix


static void
inject_params ()
{
    for (size_t p = 0;  p < fparams.size();  p += 2) {
        fparamdata[fparamindex] = atof (fparams[p+1].c_str());
        shadingsys->Parameter (fparams[p].c_str(), TypeDesc::TypeFloat,
                               &fparamdata[fparamindex]);
        fparamindex += 1;
    }
    for (size_t p = 0;  p < iparams.size();  p += 2) {
        iparamdata[iparamindex] = atoi (iparams[p+1].c_str());
        shadingsys->Parameter (iparams[p].c_str(), TypeDesc::TypeInt,
                               &iparamdata[iparamindex]);
        iparamindex += 1;
    }
    for (size_t p = 0;  p < vparams.size();  p += 4) {
        fparamdata[fparamindex+0] = atof (vparams[p+1].c_str());
        fparamdata[fparamindex+1] = atof (vparams[p+2].c_str());
        fparamdata[fparamindex+2] = atof (vparams[p+3].c_str());
        shadingsys->Parameter (vparams[p].c_str(), TypeDesc::TypeVector,
                               &fparamdata[fparamindex]);
        fparamindex += 3;
    }
    for (size_t p = 0;  p < sparams.size();  p += 2) {
        sparamdata[sparamindex] = ustring (sparams[p+1]);
        shadingsys->Parameter (sparams[p].c_str(), TypeDesc::TypeString,
                               &sparamdata[sparamindex]);
        sparamindex += 1;
    }
}



static int
add_shader (int argc, const char *argv[])
{
    shadingsys->attribute ("debug", debug2 ? 2 : (debug ? 1 : 0));
    const char *opt_env = getenv ("TESTSHADE_OPT");  // overrides opt
    if (opt_env)
        shadingsys->attribute ("optimize", atoi(opt_env));
    else if (O0 || O1 || O2)
        shadingsys->attribute ("optimize", O2 ? 2 : (O1 ? 1 : 0));
    shadingsys->attribute ("lockgeom", 1);
    shadingsys->attribute ("debugnan", debugnan);

    for (int i = 0;  i < argc;  i++) {
        inject_params ();

        shadernames.push_back (argv[i]);
        shadingsys->Shader ("surface", argv[i],
                            layername.length() ? layername.c_str() : NULL);

        layername.clear ();
        iparams.clear ();
        fparams.clear ();
        vparams.clear ();
        sparams.clear ();
    }
    return 0;
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
                "--debug", &debug, "Lots of debugging info",
                "--debug2", &debug2, "Even more debugging info",
                "--stats", &stats, "Print run statistics",
                "-g %d %d", &xres, &yres, "Make an X x Y grid of shading points",
                "-o %L %L", &outputvars, &outputfiles,
                        "Output (variable, filename)",
                "-od %s", &dataformatname, "Set the output data format to one of: "
                        "uint8, half, float",
                "--layer %s", &layername, "Set next layer name",
                "--fparam %L %L",
                        &fparams, &fparams,
                        "Add a float param (args: name value)",
                "--iparam %L %L",
                        &iparams, &iparams,
                        "Add an integer param (args: name value)",
                "--vparam %L %L %L %L",
                        &vparams, &vparams, &vparams, &vparams,
                        "Add a vector or color param (args: name x y z)",
                "--sparam %L %L",
                        &sparams, &sparams,
                        "Add a string param (args: name value)",
                "--connect %L %L %L %L",
                    &connections, &connections, &connections, &connections,
                    "Connect fromlayer fromoutput tolayer toinput",
                "--raytype %s", &raytype, "Set the raytype",
                "--iters %d", &iters, "Number of iterations",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--center", &pixelcenters, "Shade at output pixel 'centers' rather than corners",
                "--debugnan", &debugnan, "Turn on 'debugnan' mode",
                "--options %s", &extraoptions, "Set extra OSL options",
//                "-v", &verbose, "Verbose output",
                NULL);
    if (ap.parse(argc, argv) < 0 || shadernames.empty()) {
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
                     ShadingAttribStateRef &shaderstate)
{
    // Tell the shading system which outputs we want
    if (outputvars.size()) {
        std::vector<const char *> aovnames (outputvars.size());
        for (size_t i = 0; i < outputvars.size(); ++i)
            aovnames[i] = outputvars[i].c_str();
        shadingsys->attribute ("renderer_outputs",
                               TypeDesc(TypeDesc::STRING,(int)aovnames.size()),
                               &aovnames[0]);
    }

    if (extraoptions.size())
        shadingsys->attribute ("options", extraoptions);

    ShadingContext *ctx = shadingsys->get_context ();
    // Because we can only call get_symbol on something that has been
    // set up to shade (or executed), we call execute() but tell it not
    // to actually run the shader.
    ShaderGlobals sg;
    setup_shaderglobals (sg, shadingsys, 0, 0);
    shadingsys->execute (*ctx, *shaderstate, sg, false);

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



extern "C" int
test_shade (int argc, const char *argv[])
{
    OIIO::Timer timer;

    // Create a new shading system.  We pass it the RendererServices
    // object that services callbacks from the shading system, NULL for
    // the TextureSystem (that just makes 'create' make its own TS), and
    // an error handler.
    shadingsys = ShadingSystem::create (&rend, NULL, &errhandler);

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
    //   ss->ShaderGroupBegin ();
    //   ss->Parameter ("paramname", TypeDesc paramtype, void *value);
    //      ... and so on for all the other parameters of...
    //   ss->Shader ("shadertype", "shadername", "layername");
    //      The Shader() call creates a new instance, which gets
    //      all the pending Parameter() values made right before it.
    //   ... and other shader instances in this group, interspersed with...
    //   ss->ConnectShaders ("layer1", "param1", "layer2", "param2");
    //   ... and other connections ...
    //   ss->ShaderGroupEnd ();
    //   // and now grab an opaque reference to that shader group:
    //   ShadingAttribStateRef shaderstate = s->state ();
    // 
    // It looks so simple, and it really is, except that the way this
    // testshade program works is that all the Parameter() and Shader()
    // calls are done inside getargs(), as it walks through the command
    // line arguments, whereas the connections accumulate and have
    // to be processed at the end.  Bear with us.
    
    // Start the shader group.
    shadingsys->ShaderGroupBegin ();
    // Get the command line arguments.  That will set up all the shader
    // instances and their parameters for the group.
    getargs (argc, argv);

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

    // Now we should have a valid shading state, to get a reference to it.
    ShadingAttribStateRef shaderstate = shadingsys->state ();
    if (outputfiles.size() != 0)
        std::cout << "\n";

    // Set up the named transformations, including shader and object.
    // For this test application, we just do this statically; in a real
    // renderer, the global named space (like "myspace") would probably
    // be static, but shader and object spaces may be different for each
    // object.
    setup_transformations (rend, Mshad, Mobj);

    // Set up the image outputs requested on the command line
    setup_output_images (shadingsys, shaderstate);

    // Set up shader globals and a little test grid of points to shade.
    ShaderGlobals shaderglobals;

    double setuptime = timer.lap ();

    std::vector<float> pixel;

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

    // Allow a settable number of iterations to "render" the whole image,
    // which is useful for time trials of things that would be too quick
    // to accurately time for a single iteration
    for (int iter = 0;  iter < iters;  ++iter) {

        // Loop over all pixels in the image (in x and y)...
        for (int y = 0, n = 0;  y < yres;  ++y) {
            for (int x = 0;  x < xres;  ++x, ++n) {

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
                shadingsys->execute (*ctx, *shaderstate, shaderglobals);

                // Save all the designated outputs.  But only do so if we
                // are on the last iteration requested, so that if we are
                // doing a bunch of iterations for time trials, we only
                // including the output pixel copying once in the timing.
                if (iter == (iters - 1)) {
                    save_outputs (shadingsys, ctx, x, y);
                }
            }
        }
    }

    // We're done shading with this context.
    shadingsys->release_context (ctx);

    // Now that we're done rendering, release the thread=specific
    // pointer we saved.  A simple app could skip this; but if the app
    // asks for it (as we have in this example), then it should also
    // destroy it when done with it.
    shadingsys->destroy_thread_info(thread_info);

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
        double runtime = timer();
        std::cout << "\n";
        std::cout << "Setup: " << OIIO::Strutil::timeintervalformat (setuptime,2) << "\n";
        std::cout << "Run  : " << OIIO::Strutil::timeintervalformat (runtime,2) << "\n";
        std::cout << "\n";
        std::cout << shadingsys->getstats (5) << "\n";
    }

    // We're done with the shading system now, destroy it
    ShadingSystem::destroy (shadingsys);

    return EXIT_SUCCESS;
}
