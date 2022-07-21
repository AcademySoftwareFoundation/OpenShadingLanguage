// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


/*

This is an example of using OSL in batch mode to perform SIMD calculations
on a bunch of points, perhaps as you would to run a deformer.

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


To build (using CMake >= 3.13, which has -B and -S):

    cmake -S . -B build
    cmake --build build

Note that you must have OSL_ROOT set to an installed OSL in order for CMake
to find it properly.

To run:

    build/bin/osldeformer

NOTE: keep in changes sync with the scalar version testsuite/example-deformer
*/

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslexec.h>
#include <OSL/oslquery.h>
#include <OSL/rendererservices.h>

#include <OSL/batched_rendererservices.h>


// Define a userdata structure that holds any varying per-point values that
// might be retrieved by shader (interpolated data, etc.). This example is
// hard-coded, but you could also imagine that if it wasn't a fixed set
// of values, it could hold dynamic data like a name->value hash table
// or something. This is a stand-in for prim-specific or interpolatd data.
struct MyUserData {
    float amplitude = 0.0f;

#if 0
    // OBSOLETE: This is the old way.
    // Make a retrieve-by-name function.
    template<int WidthT>
    bool retrieve(OSL::ustring name, OSL::MaskedData<WidthT> wdata, OSL::ActiveLane lane)
    {
        // For speed, it wants to use ustrings (special unique strings that
        // can be equality tested at the cost of a simple pointer compare
        // without needing to examine the characters). But they are
        // expensive to create, so we use static variables to hold any of
        // these constant strings.
        static OSL::ustring uamplitude("amplitude");

        // Check if the name and type is something we know how to supply.
        if (name == uamplitude && wdata.type() == OIIO::TypeFloat) {
            // Found! Fill in the value. The renderer may ask for
            // derivatives. If that's not something we know how to compute,
            // just set them to zero. Just pass the MaskedData to MaskedDx
            // and MaskedDy to access the derivatives.
            /// If derivatives is false, don't write
            // to that memory!
            OSL::Masked<float, WidthT> wval(wdata);
            wval[lane] = amplitude;
            if (wdata.has_derivs()) {
                OSL::MaskedDx<float, WidthT> wdx(wdata);
                OSL::MaskedDx<float, WidthT> wdy(wdata);
                wdx[lane] = 0.0f;
                wdy[lane] = 0.0f;
            }
            return true;
        }
        // Not a named value we know about, so just return false.
        return false;
    }
#endif
};



template<int WidthT>
class MyBatchedRendererServices final
    : public OSL::BatchedRendererServices<WidthT> {
    OSL_USING_DATA_WIDTH(WidthT);

#if 0
    // OBSOLETE: This is the old way.
    Mask get_userdata(OSL::ustring name, BatchedShaderGlobals* bsg,
                                  MaskedData wdata) override
    {
        // In this case, our implementation of get_userdata just requests
        // it from the MyUserData, which we have arranged is pointed to
        // by shaderglobals.renderstate.
        MyUserData* userdata_base = (MyUserData*)bsg->uniform.renderstate;
        Mask success(false);
        wdata.mask().foreach([=,&success](OSL::ActiveLane lane)->void {
            auto & userdata = userdata_base[lane];
            if (userdata.retrieve(name, wdata, lane)) {
                success.set_on(lane);
            }
        });
        return success;
    }
#endif

    // Explicitly let code generator know what we have or haven't overridden
    // so it can call an internal optimized version vs. a virtual function.
    bool is_overridden_get_inverse_matrix_WmWxWf() const override
    {
        return false;
    }
    bool is_overridden_get_matrix_WmWsWf() const override
    {
        return false;
    }
    bool is_overridden_get_inverse_matrix_WmsWf() const override
    {
        return false;
    }
    bool is_overridden_get_inverse_matrix_WmWsWf() const override
    {
        return false;
    }
    bool is_overridden_texture() const override
    {
        return false;
    }
    bool is_overridden_texture3d() const override
    {
        return false;
    }
    bool is_overridden_environment() const override
    {
        return false;
    }
    bool is_overridden_pointcloud_search() const override
    {
        return false;
    }
    bool is_overridden_pointcloud_get() const override
    {
        return false;
    }
    bool is_overridden_pointcloud_write() const override
    {
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
    OSL::BatchedRendererServices<16>* batched(OSL::WidthOf<16>) override
    {
        return &m_batch_16_rs;
    }
    OSL::BatchedRendererServices<8>* batched(OSL::WidthOf<8>) override
    {
        return &m_batch_8_rs;
    }

private:
    MyBatchedRendererServices<16> m_batch_16_rs;
    MyBatchedRendererServices<8> m_batch_8_rs;
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


    // For batched allow FMA if build of OSL supports it
    const int llvm_jit_fma = 1;
    shadsys->attribute("llvm_jit_fma", llvm_jit_fma);

    // build searchpath for ISA specific OSL shared libraries based on expected
    // location of library directories relative to the executables path.
    // Users can overide using the "options" command line option
    // with "searchpath:library"
    static const char* relative_lib_dirs[] =
#if (defined(_WIN32) || defined(_WIN64))
        { "\\..\\..\\..\\lib64", "\\..\\..\\..\\lib" };
#else
        { "/../../../lib64", "/../../../lib" };
#endif
    auto executable_directory = OIIO::Filesystem::parent_path(
        OIIO::Sysutil::this_program_path());
    int dirNum = 0;
    std::string librarypath;
    for (const char* relative_lib_dir : relative_lib_dirs) {
        if (dirNum++ > 0)
            librarypath += ":";
        librarypath += executable_directory + relative_lib_dir;
    }
    shadsys->attribute("searchpath:library", librarypath);


    int batch_width = -1;
    if (shadsys->configure_batch_execution_at(16)) {
        batch_width = 16;
    } else if (shadsys->configure_batch_execution_at(8)) {
        batch_width = 8;
    } else {
        std::cout
            << "Error:  Hardware doesn't support 8 or 16 wide SIMD or the OSL has not been configured and built with a proper USE_BATCHED."
            << std::endl;
        std::cout << "Error:  e.g.:  USE_BATCHED=b8_AVX2,b8_AVX512,b16_AVX512"
                  << std::endl;
        return -1;
    }

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

    // Use the new symlocs API to say where to place outputs
    OSL::SymLocationDesc outputs("layer3.out", OSL::TypePoint, false,
                                 OSL::SymArena::Outputs,
                                 0 /* output arena offset of "out" */,
                                 sizeof(OSL::Vec3) /* point to point stride */);
    shadsys->add_symlocs(mygroup.get(), outputs);

    // And where to place inputs
    OSL::SymLocationDesc udinputs("amplitude", OSL::TypeFloat, /*derivs=*/false,
                                  OSL::SymArena::UserData,
                                  0 /* userdata arena offset of "amplitude" */,
                                  sizeof(MyUserData) /* stride */);
    shadsys->add_symlocs(mygroup.get(), udinputs);

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
    //     (0.1*i, 0, 0)   for i in [0,20)
    // We will store the inputs in Pin[] and results in Pout[].
    const int npoints = 20;
    OSL::Vec3 Pin[npoints];
    OSL::Vec3 Pout[npoints];
    MyUserData userdata[npoints];
    for (int i = 0; i < npoints; ++i) {
        Pin[i] = OSL::Vec3(0.1f * i, 0.0f, 0.0f);
        // Fill in the userdata struct for the point
        userdata[i].amplitude = 0.0f + 1.0f * powf(i / float(npoints), 3.0f);
    }

    auto batched_shadepoints = [&](auto integral_constant_width) -> void {
        constexpr int WidthT = integral_constant_width();

        // Shade the points:
        int outter_point_index = 0;
        while (outter_point_index < npoints) {
            int batchSize = std::min(WidthT, npoints - outter_point_index);

            // First, we need a ShaderGlobals struct:
            OSL::BatchedShaderGlobals<WidthT> shaderglobals;
            OSL::Block<int, WidthT> wide_shadeindex_block;

            // Set up inputs.
            for (int bi = 0; bi < batchSize; ++bi) {
                int point_index = outter_point_index + bi;

                wide_shadeindex_block[bi] = point_index;
                // Example of initializing a global: the position, P. It just lives
                // as a hard-coded field in the ShaderGlobals itself.
                // We must populate a P for each SIMD lane using the
                // array subscript with an index in [0-WidthT)
                shaderglobals.varying.P[bi] = Pin[point_index];
            }


#if 0
            // OBSOLETE: This is the old way.
            // Make a userdata record. Make sure the shaderglobals points to it.
            // MyUserData userdata;
            shaderglobals.uniform.renderstate = &userdata[outter_point_index];

            // Example of initializing a varying or interpolated parameter. We
            // MUST have declared this as a "lockgeom=0" parameter (either in
            // the shader source itself, or when we instanced it with the
            // ShadingSystem::Parameter() call) or this won't work!
            for(int bi=0; bi < batchSize; ++bi) {
                int point_index = outter_point_index + bi;
                userdata[point_index].amplitude = 0.0f + 1.0f * powf(point_index / float(npoints), 3.0f);
            }
#endif

            // Run the shader (will automagically optimize and JIT the first
            // time it executes).
            shadsys->batched<WidthT>().execute(
                *ctx, *mygroup.get(), batchSize, wide_shadeindex_block,
                shaderglobals,
                /*userdata arena start=*/&userdata,
                /*output arena start=*/&Pout);

            // OBSOLETE: This is the old way.
            // Retrieve the result. This is fast, it's just combining the data
            // area address known by the context with the offset-within-data
            // that is known in that `outsym` we retrieved once for the group.
            // Pout[i] = *(OSL::Vec3*)shadsys->symbol_address(*ctx, outsym);

            outter_point_index += batchSize;
        }
    };

    if (batch_width == 16) {
        batched_shadepoints(std::integral_constant<int, 16> {});
    } else {
        batched_shadepoints(std::integral_constant<int, 8> {});
    }

    // Print some results to prove that we generated an expected Pout.
    for (int i = 0; i < npoints; ++i) {
        OIIO::Strutil::print(
            "{:2}: Undeformed P = ({:0.3g} {:0.3g} {:0.3g})  -->  Deformed ({:0.3g} {:0.3g} {:0.3g})\n",
            i, Pin[i][0], Pin[i][1], Pin[i][2], Pout[i][0], Pout[i][1],
            Pout[i][2]);
    }

    // All done. Release the contexts and threadinfo for each thread:
    shadsys->release_context(ctx);
    shadsys->destroy_thread_info(perthread);
}
