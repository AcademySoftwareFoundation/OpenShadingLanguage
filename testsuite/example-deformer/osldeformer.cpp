// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


/*

This is an example of using OSL to perform calculations on a bunch of
points, perhaps as you would to run a deformer.

We set up an example of the following:

* We're going to execute the shader on a collection of points.
* Each point has a position (the OSL global variable P).
* There's an input parameter `amplitude` which we presume to be a varying
  interpolated value, attribute, or something like that.
* Our shader network computes a noise value based on the position and the
  amplitude parameter, adds it to P and and stores it in an output variable
  `out`.

Thus, the point of our shader network is to look at the position P and a
userdata value `amplitude`, and produce a new output `out`.

This could easily be implemented as just one simple shader, but for
illustrative purposes we will set it up as a 3-node shader network.


To build:

    mkdir build
    cd build
    cmake --config Release ..
    cmake --build .
    cd ..

Note that you must have OSL_ROOT set to an installed OSL in order for CMake
to find it properly.

To run:

    build/bin/osldeformer

*/



#include <OSL/oslexec.h>
#include <OSL/oslquery.h>


// Define a userdata structure that holds any varying per-point values that
// might be retrieved by shader (interpolated data, etc.). This example is
// hard-coded, but you could also imagine that if it wasn't a fixed set
// of values, it could hold dynamic data like a name->value hash table
// or something. This is a stand-in for prim-specific or interpolatd data.
struct MyUserData {
    float amplitude = 0.0f;

    // Make a retrieve-by-name function.
    bool retrieve(OSL::ustring name, OSL::TypeDesc type, void* val,
                  bool derivatives)
    {
        // For speed, it wants to use ustrings (special unique strings that
        // can be equality tested at the cost of a simple pointer compare
        // without needing to examine the characters). But they are
        // expensive to create, so we use static variables to hold any of
        // these constant strings.
        static OSL::ustring uamplitude("amplitude");

        // Check if the name and type is something we know how to supply.
        if (name == uamplitude && type == OIIO::TypeFloat) {
            // Found! Fill in the value. The renderer may ask for
            // derivatives. If that's not something we know how to compute,
            // just set them to zero. They will always just be the next two
            // slots after the value. If derivatives is false, don't write
            // to that memory!
            ((float*)val)[0] = amplitude;
            if (derivatives) {
                ((float*)val)[1] = 0.0f;
                ((float*)val)[2] = 0.0f;
            }
            return true;
        }
        // Not a named value we know about, so just return false.
        return false;
    }
};



// RendererServices is the interface through which OSL requests things back
// from the app (called a "renderer", but it doesn't have to literally be
// one). The important feature we are concerned about here is that this is
// how "userdata" is retrieved. We set up a subclass that overloads
// get_userdata() to retrieve it from a per-point MyUserData that whose
// pointer is stored in shaderglobals.renderstate.
class MyRendererServices final : public OSL::RendererServices {
public:
    virtual bool get_userdata(bool derivatives, OSL::ustring name,
                              OSL::TypeDesc type, OSL::ShaderGlobals* sg,
                              void* val)
    {
        // In this case, our implementation of get_userdata just requests
        // it from the MyUserData, which we have arranged is pointed to
        // by shaderglobals.renderstate.
        MyUserData* userdata = (MyUserData*)sg->renderstate;
        return userdata ? userdata->retrieve(name, type, val, derivatives)
                        : false;
    }
};



int
main(int argc, char* argv[])
{
    // Create a shading system. Note: the constructor can override with
    // optional pointers to your custom RendererServices, TextureSystem,
    // and ErrorHandler. In this case, we do supply our own subclass of
    // RendererServices, so that we can retrieve userdata.
    MyRendererServices renderer;
    std::unique_ptr<OSL::ShadingSystem> shadsys(
        new OSL::ShadingSystem(&renderer));

    // Let's create a simple deformer as a displacement shader that
    // computes P+noise(P):
    //
    //    +-------+                    +-----+
    //    | getP  |--+---------------->| add |
    //    +-------+  |                 |     |----> out
    //               |              +->|     |
    //               |              |  +-----+
    //               |  +--------+  |
    //               +->| vfBm3d |--+
    //   amplitude----->|        |
    //                  +--------+
    //
    // Of course you'd never split this simple functionality into three
    // nodes, but I am trying to illustrate how to set up the multi-node
    // case.

    OSL::ShaderGroupRef mygroup = shadsys->ShaderGroupBegin(
        "my_noise_deformer");
    shadsys->Shader(*mygroup, "displacement", "getP", "layer1");
    shadsys->Parameter(*mygroup, "amplitude", 1.0f, /*lockgeom=*/false);
    shadsys->Shader(*mygroup, "displacement", "vfBm3d", "layer2");
    shadsys->ConnectShaders(*mygroup, "layer1", "out", "layer2", "position");
    shadsys->Shader(*mygroup, "displacement", "vadd", "layer3");
    shadsys->ConnectShaders(*mygroup, "layer1", "out", "layer3", "a");
    shadsys->ConnectShaders(*mygroup, "layer2", "out", "layer3", "b");
    shadsys->ShaderGroupEnd(*mygroup);

    // Note: we could have used a serialized version equivalently:
    //
    //     std::string serialized =
    //         "shader getP layer1 ;\n"
    //         "param float amplitude 1.0 [[ int lockgeom = 0 ]];\n"
    //         "shader vfBm3d layer2 ;\n"
    //         "connect layer1.out layer2.position ;\n"
    //         "shader vadd layer3 ;\n"
    //         "connect layer1.out layer3.a ;\n"
    //         "connect layer2.out layer3.b ;\n"
    //     mygroup = shadsys->ShaderGroupBegin ("my_noise_addr", "displacement",
    //                                          serialized);
    //     shadsys->ShaderGroupEnd (*mygroup);
    //
    // Also, if we had those commands in a text file, let's say it is called
    // "material.oslgroup", then we can also specify it by filename:
    //
    //     mygroup = shadsys->ShaderGroupBegin ("my_noise_addr", "displacement",
    //                                          "material.oslgroup");
    //     shadsys->ShaderGroupEnd (*mygroup);


    // Tell the shading system the list of preserved outputs for this group.
    // Pass as an array of char*'s.
    // Note that shader outputs that are not listed here and not connected
    // to downstream layers will be optimized away, so this is important!
#if 0
    // This just adds one manually
    const char* output_names[] = { "out" };
    shadsys->attribute (mygroup.get(), "renderer_outputs",
                        OSL::TypeDesc(OSL::TypeDesc::STRING,1),
                        &output_names);
#else
    // This fancier version adds all output parameters of the final layer
    // as renderer outputs. This may not be what you want! But we're doing
    // it just to show how it can be done with oslquery:
    {
        int numlayers = 1;
        shadsys->getattribute(mygroup.get(), "num_layers", numlayers);
        std::vector<OSL::ustring> output_names;
        OSL::OSLQuery oslquery(mygroup.get(), numlayers - 1);
        for (size_t i = 0; i < oslquery.nparams(); ++i) {
            auto p = oslquery.getparam(i);
            if (p && p->isoutput)
                output_names.push_back(p->name);
        }
        shadsys->attribute(mygroup.get(), "renderer_outputs",
                           OSL::TypeDesc(OSL::TypeDesc::STRING,
                                         output_names.size()),
                           output_names.data());
    }
#endif


    // Now we want to create a context in which we can execute the shader.
    // We need one context per thread. A context can be used over and over
    // to shade multiple points, but it should never be used by more than
    // one thread (and thus, one context cannot be used to run two shader
    // executions simultaneously). But you are free to make a PerThreadInfo
    // and a ShadingContext for *each* thread, and those separate contexts
    // may execute concurrently. For this sample program, though, we will
    // only use one thread, and thus need only one context.
    OSL::PerThreadInfo* perthread = shadsys->create_thread_info();
    OSL::ShadingContext* ctx      = shadsys->get_context(perthread);

    // Get a ShaderSymbol* handle to the final output we care about. This
    // will greatly speed up retrieving the value later, rather than by
    // looking it up by name on every shade.
    // The group must already be optimized before we call find_symbol,
    // so we force that to happen now.
    shadsys->optimize_group(mygroup.get(), ctx);
    const OSL::ShaderSymbol* outsym
        = shadsys->find_symbol(*mygroup.get(), OSL::ustring("layer3"),
                               OSL::ustring("out"));
    OSL_ASSERT(outsym);

    // For illustration, let's loop over running this on points
    //     (0.1*i, 0, 0)   for i in [0,20]
    for (int i = 0; i < 20; ++i) {
        // First, we need a ShaderGlobals struct:
        OSL::ShaderGlobals shaderglobals;

        // Make a userdata record. Make sure the shaderglobals points to it.
        MyUserData userdata;
        shaderglobals.renderstate = &userdata;

        // Set up inputs.

        // Example of initializing a global: the position, P. It just lives
        // as a hard-coded field in the ShaderGlobals itself.
        OSL::Vec3 Pin(0.1f * i, 0.0f, 0.0f);
        shaderglobals.P = Pin;

        // Example of initializing a varying or interpolated parameter. We
        // MUST have declared this as a "lockgeom=0" parameter (either in
        // the shader source itself, or when we instanced it with the
        // ShadingSystem::Parameter() call) or this won't work!
        userdata.amplitude = 0.0f + 1.0f * powf(i / 20.0f, 3.0f);

        // Run the shader (will automagically optimize and JIT the first
        // time it executes).
        shadsys->execute(ctx, *mygroup.get(), shaderglobals);

        // Retrieve the result. This is fast, it's just combining the data
        // area address known by the context with the offset-within-data
        // that is known in that `outsym` we retrieved once for the group.
        OSL::Vec3 Pout = *(OSL::Vec3*)shadsys->symbol_address(*ctx, outsym);

        // Print some results to prove that we generated an expected Pout.
        std::cout << "i = " << i << "\n";
        std::cout << "Undeformed P = " << Pin
                  << "  amplitude = " << userdata.amplitude << "\n";
        std::cout << "Deformed " << Pin << "  -->  " << Pout << "\n";
        std::cout << "\n";
    }

    // All done. Release the contexts and threadinfo for each thread:
    shadsys->release_context(ctx);
    shadsys->destroy_thread_info(perthread);
}
