/*
Copyright (c) 2009-2011 Sony Pictures Imageworks Inc., et al.
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

#include <string>
#include <iostream>
#include <cstdlib>
#include <dlfcn.h>

#include <OpenImageIO/plugin.h>
using namespace OIIO;


typedef int (*EntryPoint)(int argc, const char *argv[]);



int
main (int argc, const char *argv[])
{
#ifdef __linux__
    // On Linux, if an app loads a plugin (DSO/DLL) using dlopen without
    // passing RTLD_GLOBAL as the mode, and that plugin accesses OSL (by
    // linking against liboslexec), it turns out that LLVM can fail in
    // finding symbols when it does its dlsym to resolve functions in
    // the app called by the IR.  Since we can't control the app
    // (Houdini and Maya, I'm talking to you!), we compensate by asking
    // to dlopen it here, the "right" way, and then later on, LLVM will
    // be able to find the symbols.  I haven't can't seem to reproduce
    // this issue on OS X, so for now we only do this on Linux.
    //
    // This would not be necessary if we didn't specifically disallow
    // RTLD_GLOBAL in the Plugin::open call below, but we are purposely
    // disallowing global symbol visibility in order to test this
    // solution.
    //
    // Note also that in the real world, you'd need to be really sure
    // that the file you dlopen'ed is the name "our" .so file -- beware
    // renaming.
    dlopen ("liboslexec.so", RTLD_LAZY | RTLD_GLOBAL);
#endif


    std::string pluginname = std::string("libtestshade.") 
                             + Plugin::plugin_extension();
#if OPENIMAGEIO_VERSION >= 1000 /* 0.10.0 */
    Plugin::Handle handle = Plugin::open (pluginname, 
                                          false /* NOT RTLD_GLOBAL! */);
#else
    Plugin::Handle handle = Plugin::open (pluginname);
#endif
    if (! handle) {
        std::cerr << "Could not open " << pluginname << "\n";
        exit (1);
    }

    EntryPoint entry = (EntryPoint) Plugin::getsym (handle, "test_shade");
    if (! entry) {
        std::cerr << "Cound not find test_shade symbol\n";
        exit (1);
    }

    int r = entry (argc, argv);

    Plugin::close (handle);
    return r;
}

