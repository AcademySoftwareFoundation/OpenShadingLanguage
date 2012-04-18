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

#include <vector>
#include <string>
#include <fstream>
#include <cstdio>

#include "osoreader.h"

#define yyFlexLexer osoFlexLexer
#include "FlexLexer.h"

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"


OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


osoFlexLexer * OSOReader::osolexer = NULL;
OSOReader * OSOReader::osoreader = NULL;
OIIO::mutex OSOReader::m_osoread_mutex;



bool
OSOReader::parse (const std::string &filename)
{
    // The lexer/parser isn't thread-safe, so make sure Only one thread
    // can actually be reading a .oso file at a time.
    OIIO::lock_guard guard (m_osoread_mutex);

    std::fstream input (filename.c_str(), std::ios::in);
    if (! input.is_open()) {
        m_err.error ("File %s not found", filename.c_str());
        return false;
    }

    osoreader = this;
    osolexer = new osoFlexLexer (&input);
    assert (osolexer);
    bool ok = ! osoparse ();   // osoparse returns nonzero if error
    if (ok) {
//        m_err.info ("Correctly parsed %s", filename.c_str());
    } else {
        m_err.error ("Failed parse of %s", filename.c_str());
    }
    delete osolexer;
    osolexer = NULL;

    input.close ();
    return ok;
}



}; // namespace pvt
OSL_NAMESPACE_EXIT
