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

#ifdef __linux__
#include <dlfcn.h>
#endif

#include <OpenImageIO/plugin.h>
using namespace OIIO;


typedef int (*EntryPoint)(int argc, const char *argv[]);



int
main (int argc, const char *argv[])
{
    std::string pluginname = std::string("libtestshade.") 
                             + Plugin::plugin_extension();
    Plugin::Handle handle = Plugin::open (pluginname, 
                                          false /* NOT RTLD_GLOBAL! */);
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

