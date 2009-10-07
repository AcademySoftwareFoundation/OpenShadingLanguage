/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include "oslops.h"
#include "oslexec_pvt.h"

#include "OpenImageIO/sysutil.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {



DECLOP (OP_texture)
{
    // Grab the required arguments: result, filename, s, t
    DASSERT (nargs >= 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Filename (exec->sym (args[1]));
    DASSERT (Filename.typespec().is_string());
    Symbol &S (exec->sym (args[2]));
    Symbol &T (exec->sym (args[3]));
    DASSERT (S.typespec().is_float() && T.typespec().is_float());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* Assume texture always varies */);

    // FIXME -- we should allow derivs of texture
    if (Result.has_derivs ())
        exec->zero_derivs (Result);

    VaryingRef<float> result ((float *)Result.data(), Result.step());
    VaryingRef<float> s ((float *)S.data(), S.step());
    VaryingRef<float> t ((float *)T.data(), T.step());
    VaryingRef<ustring> filename ((ustring *)Filename.data(), Filename.step());

    TextureSystem *texturesys = exec->texturesys ();
    TextureOptions options;

    // Set up derivs
    float zero = 0.0f;
    VaryingRef<float> dsdx, dsdy, dtdx, dtdy;
    if (S.has_derivs()) {
        dsdx.init ((float *)S.data() + 1, S.step());
        dsdy.init ((float *)S.data() + 2, S.step());
    } else {
        dsdx.init (&zero);
        dsdy.init (&zero);
    }
    if (T.has_derivs()) {
        dtdx.init ((float *)T.data() + 1, T.step());
        dtdy.init ((float *)T.data() + 2, T.step());
    } else {
        dtdx.init (&zero);
        dtdy.init (&zero);
    }

    // Parse all the optional arguments
    for (int a = 4;  a < nargs;  ++a) {
        Symbol &Name (exec->sym (args[a]));
        DASSERT (Name.typespec().is_string() &&
                 "optional texture token must be a string");
        DASSERT (a+1 < nargs && "malformed argument list for texture");
        if (Name.is_varying()) {
            exec->warning ("optional texture argument is a varying string! Seems pretty fishy.");
        }
        ++a;  // advance to next argument
        Symbol &Val (exec->sym (args[a]));
        ustring name = * (ustring *) Name.data();
        if (name == Strings::width) {
            options.swidth.init ((float *)Val.data(), Val.step());
            options.twidth.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::swidth) {
            options.swidth.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::twidth) {
            options.twidth.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::blur) {
            options.sblur.init ((float *)Val.data(), Val.step());
            options.tblur.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::sblur) {
            options.sblur.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::tblur) {
            options.tblur.init ((float *)Val.data(), Val.step());
        }
    }

    options.nchannels = (Result.typespec().is_triple() ? 3 : 1);
    float *r = &result[0];
    bool tempresult = false;
    if (Result.has_derivs()) {
        tempresult = true;
        r = ALLOCA (float, endpoint*options.nchannels);
    }
    for (int i = beginpoint;  i < endpoint;  ++i) {
        // FIXME -- this calls texture system separately for each point!
        // We really want to batch it into groups that share the same texture
        // filename.
        if (runflags[i]) {
#if 0
            // For comparison: one-point texture call
            bool ok = texturesys->texture (filename[i], options, 
                                           s[i], t[i], dsdx[i], dtdx[i], dsdy[i], dtdy[i],
                                           &r[i*options.nchannels]);
#else
            bool ok = texturesys->texture (filename[i], options, 
                                           runflags, i /*beginpoint*/, i+1 /*endpoint*/,
                                           s, t, dsdx, dtdx, dsdy, dtdy,
                                           r);
#endif
            if (! ok) {
                std::string err = texturesys->geterror ();
                exec->error ("%s", err.c_str());
            }
         }
    }

    if (Result.has_derivs()) {
        for (int i = beginpoint;  i < endpoint;  ++i) {
            if (runflags[i])
                for (int c = 0;  c < options.nchannels;  ++c)
                    (&result[i])[c] = r[i*options.nchannels+c];
        }
    }
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
